#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import functools
import os, shutil
import numpy as np

import torch
import copy
import time
import math
import os
import itertools
import warnings
import torch
import random
import copy
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from gpu import add_gpu_params, parse_gpu, distributed_opt, distributed_sync, cleanup
from sfl.model.attacker.sip.inversion_models import get_inverter_with_config
from sfl.utils.model import calc_unshift_loss, evaluate_attacker_rouge
from data_utils import FT_Dataset
# from splitmodel import GPT2Config, GPT2LMModel_Server, GPT2LMModel_Client
from U_shaped_splitmodel import GPT2Config, GPT2LMModel_Head, GPT2LMModel_Server, GPT2LMModel_Tail
from options import args_parser
import loralib as lora
from torch.optim import Adam, AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
from sfl.utils.exp import get_dataset, print_args


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))


def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))


def print_trainable_parameters(*models):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for idx, model in enumerate(models):
        for name, param in model.named_parameters():
            num_params = param.numel()  # 当前参数的总数
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                non_trainable_params += num_params

    # 打印结果
    print("=" * 50)
    print(f"  Total Parameters: {total_params}")
    print(f"  Trainable Parameters: {trainable_params}")
    print(f"  Non-trainable Parameters: {non_trainable_params}")
    print("=" * 50)


def inspect_model_parameters(model):
    """
    打印模型中参数的名称和是否可训练的信息。

    Args:
        model (torch.nn.Module): PyTorch模型对象
    """
    for name, param in model.named_parameters():
        print(f"Layer Name: {name}, Trainable: {param.requires_grad}")


def print_args(args):
    if args.rank == 0:
        print("=" * 100)
        for k, v in args.__dict__.items():
            print(f"        - {k} : {v}")
        print("=" * 100)


def save_checkpoint(w_glob_head, model_server, w_glob_tail, args, train_step, num_clients):
    if args.rank != 0:
        return

    model_state_dict = {}

    # rename the key in client model
    for key, value in w_glob_head.items():
        new_key = ""
        if key.startswith("transformer_Head"):
            new_key = key.replace("transformer_Head", "module.transformer")
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    # rename the key in server model
    for key, value in model_server.state_dict().items():
        new_key = ""
        # print(key)
        if key.startswith("module.transformer_Server"):
            new_key = key.replace("module.transformer_Server", "module.transformer")
        else:
            print(key)
            model_state_dict[key] = value

        if new_key.startswith("module.transformer.h."):
            parts = key.split(".")
            layer_idx = int(parts[3])
            new_key = ".".join(["module.transformer.h", str(layer_idx + 3)] + parts[4:])
            model_state_dict[new_key] = value
        else:
            model_state_dict[new_key] = value

    # rename the key in client model
    for key, value in w_glob_tail.items():
        new_key = ""
        if key.startswith("transformer_Tail"):
            new_key = key.replace("transformer_Tail", "module.transformer")
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value

    if args.model_card == "gpt2.md":
        model_path = os.path.join(
            "./trained_models/GPT2_M/e2e",
            f"model_sfl.{train_step}_r={args.lora_dim}_num={num_clients}_block=3_seed={args.random_seed}.pt",
        )
    if args.model_card == "gpt2.sm":
        model_path = os.path.join(
            "./trained_models/GPT2_S/e2e",
            f"model_sfl.{train_step}_r={args.lora_dim}_num={num_clients}_block=3_seed={args.random_seed}.pt",
        )
    print("saving checkpoint", model_path)
    torch.save({"model_state_dict": model_state_dict}, model_path)


def fed_avg(w): # weighted average of the models
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


class AverageMeter(object): # 
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def optimizer_step( 
    _loss,
    optimizer_server,
    model_server,
    optimizer_head,
    optimizer_tail,
    _schedule,
    head_hidden_states,
    hidden_states0,
    server_hidden_states,
    hidden_states1,
    args,
    is_update,
):
    if args.fp16:
        with amp.scale_loss(_loss, optimizer_server) as _scaled_loss:
            _scaled_loss.backward()
    else:
        _loss.backward()

    dfx_server = server_hidden_states.grad.clone().detach() 
     
    if is_update and args.clip > 0: # clip the gradient if necessary
        if args.fp16:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer_server), args.clip
            )
        else:
            torch.nn.utils.clip_grad_norm_(model_server.parameters(), args.clip)

    optimizer_tail.step()
    optimizer_tail.zero_grad()

    hidden_states1.backward(dfx_server)
    dfx_client = head_hidden_states.grad.clone().detach()

    optimizer_server.step()
    optimizer_server.zero_grad()

    if _schedule is not None: # update learning rate scheduler
        _schedule.step()

    # hidden_states0.backward(dfx_client)
    # optimizer_head.step()
    # optimizer_head.zero_grad()


def evaluate(model_head, model_server, model_tail, valid_loader,args):
    model_head.eval()
    model_server.eval()
    model_tail.eval()

    device = torch.device("cuda")
    model_server = model_server.to(device)

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):
            _input = batch["input_ids"].to(device)
            _target = batch["input_ids"]
            _msk = batch["attention_mask"]

            hidden_states0, head_presents, _ = model_head(_input)

            hidden_states1, server_presents, _ = model_server(_input.shape, hidden_states0, head_presents)

            _, _loss, _ = model_tail(
                _input.shape,
                hidden_states1,
                server_presents,
                lm_labels=_target,
                lm_mask=_msk,
            )

            loss = _loss

            avg_lm_loss.update(loss.item())

            if idx % 100 == 0:
                print("eval samples:", idx, "loss:", loss.float())

        print("average loss", avg_lm_loss.avg)
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model_head,
    model_server,
    model_tail,
    client_head_models,
    client_tail_models,
    head_optimizers,
    tail_optimizers,
    optimizer_server,
    scheduler_server,
    train_loader,
    test_loader,
    args,
    config,
    train_step=0,
    epoch=0,
    
):
    """
    Function to train and validate federated learning models.

    Args:
        model_Client (torch.nn.Module): Client-side GPT-2 model.
        model_Server (torch.nn.Module): Server-side GPT-2 model.
        client_models (list): List of client GPT-2 models.
        optimizers (list): Optimizers for client models.
        optimizer_Server (torch.optim.Optimizer): Optimizer for server model.
        scheduler_Server (torch.optim.lr_scheduler): Learning rate scheduler for server optimizer.
        train_loader0, train_loader1, train_loader2 (torch.utils.data.DataLoader): Train loaders for three clients.
        valid_loader (torch.utils.data.DataLoader): Validation loader.
        args (argparse.Namespace): Command-line arguments.
        train_step (int, optional): Current training step. Default is 0.
        epoch (int, optional): Current epoch. Default is 0.

    Returns:
        Train step
    """
    model_head.train()
    model_server.train()
    model_tail.train()
    # Meter to average language model loss
    avg_lm_loss = AverageMeter()
    print("start to train the model................", epoch)
    log_start_time = time.time()

    # Meter to average language model loss
    best_val_ppl = None

    device = torch.device("cuda")
    # train_loader.sampler.set_epoch(epoch) # set the epoch for train_loader

    # Initialize global client model
    net_glob_head = GPT2LMModel_Head(config).to(device)
    net_glob_tail = GPT2LMModel_Tail(config).to(device)

    state_dict = torch.load(args.init_checkpoint)

    # Load weights to global client model
    net_glob_head.load_weight(state_dict)
    net_glob_tail.load_weight(state_dict)
    if args.lora_dim > 0: # mark only lora-values as trainable
        lora.mark_only_lora_as_trainable(net_glob_head)

    net_glob_head.train()
    net_glob_tail.train()
    w_glob_head = net_glob_head.state_dict() # get the global client model weights
    w_glob_tail = net_glob_tail.state_dict()
    # aggregate every 100 train_step
    aggregate_step = 100

    w_locals_head = []
    w_locals_tail = []
    log_list = []

    # get train data from different client train dataset
    
    for idx, data in enumerate(train_loader):
        # The client interacts with the server in turn
        for i in range(args.num_clients):
            client_data = data

            _input = client_data["input_ids"].to(device)
            _target = client_data["input_ids"]
            _msk = client_data["attention_mask"]

            client_head_models[i].train()

            hidden_states0, head_presents, w_head = client_head_models[i](_input)
            train_step += 1

            head_hidden_states = hidden_states0.clone().detach().requires_grad_(True) # 

            hidden_states1, server_presents, w_server = model_server(
                _input.shape,
                head_hidden_states,
                head_presents,
            )

            server_hidden_states = hidden_states1.clone().detach().requires_grad_(True)

            _, _lm_loss, w_tail = client_tail_models[i](
                _input.shape,
                server_hidden_states,
                server_presents,
                lm_labels=_target,
                lm_mask=_msk,
                label_smooth=args.label_smooth,
            )

            # _lm_loss = _lm_loss.mean()

            if (train_step + args.num_clients) % aggregate_step <= args.num_clients: # if the train_step is a multiple of aggregate_step, aggregate the client model
                w_locals_head.append(copy.deepcopy(w_head))
                w_locals_tail.append(copy.deepcopy(w_tail))

            is_update = train_step % args.grad_acc == 0  # whether to update the model
            avg_lm_loss.update(_lm_loss.item())

            optimizer_step(
                _loss=_lm_loss / args.grad_acc,
                optimizer_server=optimizer_server,
                model_server=model_server,
                optimizer_head=head_optimizers[i],
                optimizer_tail=tail_optimizers[i],
                _schedule=scheduler_server,
                head_hidden_states=head_hidden_states,
                hidden_states0=hidden_states0,
                server_hidden_states=server_hidden_states,
                hidden_states1=hidden_states1,
                args=args,
                is_update=is_update,
            )

            # aggregate client LoRA model every 100 train_step
            if train_step % aggregate_step == 0:
                temp_dict = {}
                w_locals_head_lora = []  # only aggregate lora-values

                for w_head in w_locals_head:
                    for key, value in w_head.items():
                        if key.endswith("lora_A"):
                            temp_dict[key] = value
                        if key.endswith("lora_B"):
                            temp_dict[key] = value
                    w_locals_head_lora.append(copy.deepcopy(temp_dict))

                w_glob_head_lora = fed_avg(w_locals_head_lora)

                w_glob_head_lora_new = {}

                for key, value in w_glob_head_lora.items():
                    new_key = "transformer_Head." + key
                    w_glob_head_lora_new[new_key] = value

                for key, value in w_glob_head.items():
                    if key.endswith("lora_A"):
                        w_glob_head[key] = w_glob_head_lora_new[key]
                    if key.endswith("lora_B"):
                        w_glob_head[key] = w_glob_head_lora_new[key]

                net_glob_head.load_state_dict(w_glob_head)
                for head_model in client_head_models:
                    head_model.load_state_dict(w_glob_head)

                w_locals_head = []

                temp_dict = {}
                w_locals_tail_lora = []  # only aggregate lora-values

                for w_tail in w_locals_tail:
                    for key, value in w_tail.items():
                        if key.endswith("lora_A"):
                            temp_dict[key] = value
                        if key.endswith("lora_B"):
                            temp_dict[key] = value
                    w_locals_tail_lora.append(copy.deepcopy(temp_dict))

                w_glob_tail_lora = fed_avg(w_locals_tail_lora)

                w_glob_tail_lora_new = {}

                for key, value in w_glob_tail_lora.items():
                    new_key = "transformer_Tail." + key
                    w_glob_tail_lora_new[new_key] = value

                for key, value in w_glob_tail.items():
                    if key.endswith("lora_A"):
                        w_glob_tail[key] = w_glob_tail_lora_new[key]
                    if key.endswith("lora_B"):
                        w_glob_tail[key] = w_glob_tail_lora_new[key]

                net_glob_tail.load_state_dict(w_glob_tail)
                for tail_model in client_tail_models:
                    tail_model.load_state_dict(w_glob_tail)

                w_locals_tail = []

            # Output the training process data
            if train_step % args.log_interval == 0:
                elapsed = time.time() - log_start_time
                lr = optimizer_server.param_groups[0]["lr"]
                log_str = (
                    f"| epoch {epoch:3d} step {train_step:>8d} | {idx*3 + 1:>6d} batches | "
                    f"lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | "
                    f"loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | "
                    f"ppl {math.exp(avg_lm_loss.avg):5.2f}"
                )

                log_list.append(log_str)

                if args.rank == 0:
                    print(log_str)
                log_start_time = time.time()
                avg_lm_loss.reset()

            # save checkpoint at each save_interval
            if train_step % args.save_interval == 0:
                save_checkpoint(
                    w_glob_head, model_server, w_glob_tail, args, train_step, args.num_clients
                )
            distributed_sync(args)

            if train_step % args.eval_interval == 0:
                eval_start_time = time.time()

                valid_loss, valid_ppl = evaluate(
                    net_glob_head, model_server, net_glob_tail, test_loader, args
                )
                if best_val_ppl is None or valid_ppl < best_val_ppl:
                    best_val_ppl = valid_ppl

                log_str = (
                    f"| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | "
                    f"time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | "
                    f"valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} "
                )
                log_list.append(log_str)

                if args.rank == 0:
                    print("-" * 100)
                    print(log_str)
                    print("-" * 100)

                net_glob_head.train()
                model_server.train()
                net_glob_tail.train()
                distributed_sync(args)

            # Save training process
            if train_step == args.max_step:
                df = pd.DataFrame(log_list, columns=["Log"])
                df.to_excel(
                    f"{args.model_card} rank={args.lora_dim} num={args.num_clients} block=3 seed={args.random_seed}.xlsx",
                    sheet_name="Sheet1",
                    index=False,
                )
                break

    # Save the final checkpoint
    if train_step == args.max_step:
        save_checkpoint(w_glob_head, model_server, w_glob_tail, args, train_step, args.num_clients)
    distributed_sync(args)
    return train_step, net_glob_head, model_server, net_glob_tail


def llm_head_forward(model, batch, tokenizer):
    if model.type == 'encoder-decoder':
        pass
        # input_args = get_t5_input(batch, tokenizer, model.device)
        # enc_hidden, dec_hidden = model(**input_args)
        # intermediate = torch.concat([enc_hidden, dec_hidden], dim=1)
        # input_ids = torch.concat([input_args['input_ids'], input_args['decoder_input_ids']],
        #                          dim=1).to(model.device)
    else:
        input_ids = batch['input_ids'].cuda()
        intermediate, _, _ = model(input_ids=input_ids)
    return input_ids, intermediate


def SIPAttackTraining(config, SIP_train_args, lm_net_Head):
    tokenizer = AutoTokenizer.from_pretrained("/home/lzh/projects/privacy-split-llm/data/models/gpt2-large", trust_remote_code=True)
    SIP_dataset = get_dataset('wikitext', tokenizer=tokenizer, client_ids=[])
    SIP_train_loader = SIP_dataset.get_dataloader_unsliced(SIP_train_args.batch_size, 'train')

    inverter_clz, cfg = get_inverter_with_config('gru')
    cfg.hidden_size = 256
    cfg.n_embed = config.n_embd
    cfg.vocab_size = tokenizer.vocab_size
    cfg.dropout = 0.1
    cfg.target_model = 'gpt2-large'
    attack_model = inverter_clz(cfg, reduce_dim=None)

    print(attack_model)
    opt_cls = AdamW
    optimizer = opt_cls(attack_model.parameters(), lr=SIP_train_args.lr, weight_decay=SIP_train_args.weight_decay)
    attack_model.cuda()
    log_strs = []

    with tqdm(total=SIP_train_args.epochs * len(SIP_train_loader)) as pbar:
        for epc in range(SIP_train_args.epochs):
            lm_net_Head.train(True)
            rouge_l_f = 0
            item_count = 0
            for step, batch in enumerate(SIP_train_loader):
                optimizer.zero_grad()
                input_ids, intermediate = llm_head_forward(lm_net_Head, batch, tokenizer) # 得到输入和中间层结果
                logits = attack_model(intermediate)
                loss = calc_unshift_loss(logits, input_ids)
                loss.backward()
                optimizer.step()
                res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
                rouge_l_f += res['rouge-l']['f']

                pbar.set_description(
                    f'Epoch {epc} | Loss {loss.item():.5f} | Rouge_Lf1 {rouge_l_f / (step + 1):.4f}')
                pbar.update(1)
                item_count += 1
            log_str = f"Epoch {epc} | Loss {loss.item():.5f} | Rouge_Lf1 {rouge_l_f / item_count:.4f}"
            log_strs.append(log_str)

    for log_str in log_strs:
        print(log_str)

    return attack_model

def SIPAttackEvaluate(SIP_train_args, attack_model, lm_net_Head, valid_dataset = None):
    tokenizer = AutoTokenizer.from_pretrained("/home/lzh/projects/privacy-split-llm/data/models/gpt2-large", trust_remote_code=True)
    if valid_dataset is None:
        SIP_dataset = get_dataset('wikitext', tokenizer=tokenizer, client_ids=[])
        SIP_valid_loader = SIP_dataset.get_dataloader_unsliced(SIP_train_args.batch_size, 'validation')
    else:
        tokenizer.pad_token = tokenizer.eos_token
        SIP_dataset = get_dataset(valid_dataset, tokenizer=tokenizer, client_ids=[])
        SIP_valid_loader = SIP_dataset.get_dataloader_unsliced(SIP_train_args.batch_size, 'test')

    rouge_l_f = 0  # 初始化 Rouge_Lf1
    total_steps = len(SIP_valid_loader)  # 获取总步数
    Loss = 0
    # 创建 tqdm 进度条
    with tqdm(total=total_steps, desc="Training Progress") as pbar:
        for step, batch in enumerate(SIP_valid_loader):
            attack_model.eval()
            input_ids, intermediate = llm_head_forward(lm_net_Head, batch, tokenizer)  # 得到输入和中间层结果
            logits = attack_model(intermediate)
            loss = calc_unshift_loss(logits, input_ids)
            Loss += loss.item()
            res, _, _ = evaluate_attacker_rouge(tokenizer, logits, batch)
            rouge_l_f += res['rouge-l']['f']

            # 更新 tqdm 描述信息
            pbar.set_description(
                f'Loss: {Loss / (step + 1):.5f} | Rouge_Lf1: {rouge_l_f / (step + 1):.4f}'
            )
            pbar.update(1)  
