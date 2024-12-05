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
from optimizer import (
    create_optimizer_scheduler,
    add_optimizer_params,
    create_adam_optimizer_from_args,
)

from data_utils import FT_Dataset
# from splitmodel import GPT2Config, GPT2LMModel_Server, GPT2LMModel_Client
from U_shaped_splitmodel import GPT2Config, GPT2LMModel_Head, GPT2LMModel_Server, GPT2LMModel_Tail
from exp_utils import create_exp_dir, print_trainable_parameters, inspect_model_parameters
from options import args_parser
import loralib as lora

torch.set_printoptions(threshold=100000) # set the print options for torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set the visible GPU device


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

    for key, value in w_glob_tail.items():
        new_key = ""
        if key.startswith("transformer_Tail"):
            new_key = key.replace("transformer_Tail", "module.transformer")
        else:
            model_state_dict[key] = value
        if new_key.startswith("module.transformer.h."):
            parts = key.split(".")
            layer_idx = int(parts[3])
            new_key = ".".join(["module.transformer.h", str(layer_idx + 21)] + parts[4:])
            model_state_dict[new_key] = value
        else:
            model_state_dict[new_key] = value

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

    hidden_states0.backward(dfx_client)
    optimizer_head.step()
    optimizer_head.zero_grad()


def evaluate(model_head, model_server, model_tail, valid_loader,args):
    model_head.eval()
    model_server.eval()
    model_tail.eval()

    device = torch.device("cuda")
    model_server = model_server.to(device)

    avg_lm_loss = AverageMeter()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            data = {key: value.to(device) for key, value in data.items()}

            _input = data["input"]
            _target = data["target"]
            _msk = data["mask"]

            hidden_states0, head_presents, _ = model_head(_input)

            hidden_states1, server_presents, _ = model_server(_input.shape, hidden_states0, head_presents)

            _, _loss, _ = model_tail(
                _input.shape,
                hidden_states1,
                server_presents,
                lm_labels=_target,
                lm_mask=_msk,
            )
            loss = _loss.mean()

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
    train_loader0,
    train_loader1,
    train_loader2,
    valid_loader,
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
    train_loader0.sampler.set_epoch(epoch) # set the epoch for train_loader0

    # Initialize global client model
    net_glob_head = GPT2LMModel_Head(config).to(device)
    net_glob_tail = GPT2LMModel_Tail(config).to(device)

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
    for idx, data in enumerate(zip(train_loader0, train_loader1, train_loader2)):
        # The client interacts with the server in turn
        for i in range(args.num_clients):
            client_data = {key: value.to(device) for key, value in data[i].items()}

            _input = client_data["input"].to(device)
            _target = client_data["target"]
            _msk = client_data["mask"]

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

            _lm_loss = _lm_loss.mean()

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
                    net_glob_head, model_server, net_glob_tail, valid_loader, args
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
    return train_step


if __name__ == "__main__":
    args = args_parser()
    parse_gpu(args)
    print_args(args)

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.rank == 0: # create logging directory
        args.logging = create_exp_dir(args.work_dir)

    train_data0 = FT_Dataset(
        args.train_data0,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )
    train_data1 = FT_Dataset(
        args.train_data1,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )
    train_data2 = FT_Dataset(
        args.train_data2,
        args.train_batch_size,
        args.seq_len,
        joint_lm=args.obj == "jlm",
    )
    valid_data = FT_Dataset(
        args.valid_data,
        args.valid_batch_size,
        args.seq_len,
    )
    
    train_loader0 = DataLoader(
        train_data0,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data0, seed=args.random_seed, shuffle=True
        ),
    )
    train_loader1 = DataLoader(
        train_data1,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data1, seed=args.random_seed, shuffle=True
        ),
    )
    train_loader2 = DataLoader(
        train_data2,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            train_data2, seed=args.random_seed, shuffle=True
        ),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.valid_batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        sampler=torch.utils.data.distributed.DistributedSampler(
            valid_data, seed=args.random_seed
        ),
    )

    if args.model_card == "gpt2.sm":
        config = GPT2Config(
            n_embd=768,
            n_layer=12,
            n_head=12,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == "gpt2.md":
        config = GPT2Config(
            n_embd=1024,
            n_layer=24,
            n_head=16,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == "gpt2.lg":
        config = GPT2Config(
            n_embd=1280,
            n_layer=36,
            n_head=20,
            lora_attn_dim=args.lora_dim,
            lora_attn_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    
    lm_net_Head = GPT2LMModel_Head(config)
    lm_net_Server = GPT2LMModel_Server(config)
    lm_net_Tail = GPT2LMModel_Tail(config)

    state_dict = torch.load(args.init_checkpoint)

    if args.init_checkpoint is not None: # load model weight
        lm_net_Head.load_weight(state_dict)
        lm_net_Server.load_weight(state_dict)
        lm_net_Tail.load_weight(state_dict)
        print("loaded model pretrained weight.")

    lm_net_Head = lm_net_Head.cuda()
    lm_net_Server = lm_net_Server.cuda()
    lm_net_Tail = lm_net_Tail.cuda()

    if args.lora_dim > 0: # mark lora-values as trainable
        print("marking lora-values as trainable.")
        lora.mark_only_lora_as_trainable(lm_net_Head)
        lora.mark_only_lora_as_trainable(lm_net_Server)
        lora.mark_only_lora_as_trainable(lm_net_Tail)

    print_trainable_parameters(lm_net_Head, lm_net_Server, lm_net_Tail)
    # inspect_model_parameters(lm_net_Tail)

    print("creating optimizer and scheduler.")
    optimizer_Head = create_adam_optimizer_from_args(lm_net_Head, args)
    optimizer_Server = create_adam_optimizer_from_args(lm_net_Server, args)
    optimizer_Tail = create_adam_optimizer_from_args(lm_net_Tail, args)

    # nums of clients:
    num_clients = args.num_clients

    client_head_models = []
    client_tail_models = []
    head_optimizers = []
    tail_optimizers = []

    # Create client head/tail models for different clients
    for i in range(num_clients):
        client_head_model = GPT2LMModel_Head(config)
        client_head_model.load_weight(state_dict)
        client_head_model = client_head_model.cuda()
        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(client_head_model)
        optimizer = create_adam_optimizer_from_args(client_head_model, args)
        client_head_models.append(client_head_model)
        head_optimizers.append(optimizer)

        client_tail_model = GPT2LMModel_Tail(config)
        client_tail_model.load_weight(state_dict)
        client_tail_model = client_tail_model.cuda()
        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(client_tail_model)
        optimizer = create_adam_optimizer_from_args(client_tail_model, args)
        client_tail_models.append(client_tail_model)
        tail_optimizers.append(optimizer)

    if args.max_step is None: # set max_step if not set
        args.max_step = (
            args.max_epoch * train_data0.num_batches * 3 + args.world_size - 1
        ) // args.world_size
        print("set max_step:", args.max_step)

    scheduler_Head = create_optimizer_scheduler(optimizer_Head, args)
    scheduler_Server = create_optimizer_scheduler(optimizer_Server, args)
    scheduler_Tail = create_optimizer_scheduler(optimizer_Tail, args)

    # Distributed training,s 将 model 转换为 PyTorch 的 DistributedDataParallel (DDP) 模式
    lm_net_Head, optimizer_Head = distributed_opt(
        args, lm_net_Head, optimizer_Head, grad_acc=args.grad_acc
    )
    lm_net_Server, optimizer_Server = distributed_opt(
        args, lm_net_Server, optimizer_Server, grad_acc=args.grad_acc
    )
    lm_net_Tail, optimizer_Tail = distributed_opt(
        args, lm_net_Tail, optimizer_Tail, grad_acc=args.grad_acc
    )

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                model_head=lm_net_Head,
                model_server=lm_net_Server,
                model_tail=lm_net_Tail,
                client_head_models=client_head_models,
                client_tail_models=client_tail_models,
                head_optimizers=head_optimizers,
                tail_optimizers=tail_optimizers,
                optimizer_server=optimizer_Server,
                scheduler_server=scheduler_Server,
                train_loader0=train_loader0,
                train_loader1=train_loader1,
                train_loader2=train_loader2,
                valid_loader=valid_loader,
                args=args,
                config=config,
                train_step=train_step,
                epoch=epoch,
            )
                
            if train_step >= args.max_step or (
                args.max_epoch is not None and epoch >= args.max_epoch
            ):
                if args.rank == 0:
                    print("-" * 100)
                    print("End of training")
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print("-" * 100)
            print("Exiting from training early")

    