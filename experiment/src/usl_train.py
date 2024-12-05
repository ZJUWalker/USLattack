import loralib as lora
from exp_utils import print_trainable_parameters
from optimizer import (
    create_optimizer_scheduler,
    add_optimizer_params,
    create_adam_optimizer_from_args,
)
from U_shaped_splitmodel import GPT2Config, GPT2LMModel_Head, GPT2LMModel_Server, GPT2LMModel_Tail
from gpu import add_gpu_params, parse_gpu, distributed_opt, distributed_sync, cleanup
import itertools
from exp_utils import train_validate
from sfl.utils.exp import get_dataset, print_args
from sfl.utils.model import calc_unshift_loss
from transformers import AutoTokenizer
import time
import copy

def usl_train(args, config, state_dict, lm_net_Head, lm_net_Server, lm_net_Tail, dataset):
    tokenizer = AutoTokenizer.from_pretrained("/home/lzh/projects/privacy-split-llm/data/models/gpt2-large", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    USL_dataset = get_dataset(dataset, tokenizer=tokenizer, client_ids=[])
    USL_train_loader = USL_dataset.get_dataloader_unsliced(4, 'train')
    USL_test_loader = USL_dataset.get_dataloader_unsliced(4, 'test')

    if args.lora_dim > 0: # mark lora-values as trainable
        print("marking lora-values as trainable.")
        lora.mark_only_lora_as_trainable(lm_net_Head)
        lora.mark_only_lora_as_trainable(lm_net_Server)
        lora.mark_only_lora_as_trainable(lm_net_Tail)

    print_trainable_parameters(lm_net_Head, lm_net_Server, lm_net_Tail)

        # nums of clients:
    num_clients = args.num_clients

    client_head_models = []
    client_tail_models = []
    head_optimizers = []
    tail_optimizers = []

    # Create client head/tail models for different clients
    for i in range(num_clients):
        client_head_model = copy.deepcopy(lm_net_Head)
        for param in client_head_model.parameters():
            param.requires_grad = False
        client_head_model = client_head_model.cuda()
        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(client_head_model)
        optimizer = create_adam_optimizer_from_args(client_head_model, args)
        client_head_models.append(client_head_model)
        head_optimizers.append(optimizer)

        client_tail_model = copy.deepcopy(lm_net_Tail)
        client_tail_model = client_tail_model.cuda()
        if args.lora_dim > 0:
            lora.mark_only_lora_as_trainable(client_tail_model)
        optimizer = create_adam_optimizer_from_args(client_tail_model, args)
        client_tail_models.append(client_tail_model)
        tail_optimizers.append(optimizer)

    if args.max_step is None: # set max_step if not set
        args.max_step = (
            args.max_epoch * len(USL_train_loader) + args.world_size - 1
        ) // args.world_size
        # args.max_step = 1000
        print("set max_step:", args.max_step)

    print("creating optimizer and scheduler.")
    optimizer_Head = create_adam_optimizer_from_args(lm_net_Head, args)
    optimizer_Server = create_adam_optimizer_from_args(lm_net_Server, args)
    optimizer_Tail = create_adam_optimizer_from_args(lm_net_Tail, args)

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
            train_step, net_glob_head, model_server, net_glob_tail = train_validate(
                model_head=lm_net_Head,
                model_server=lm_net_Server,
                model_tail=lm_net_Tail,
                client_head_models=client_head_models,
                client_tail_models=client_tail_models,
                head_optimizers=head_optimizers,
                tail_optimizers=tail_optimizers,
                optimizer_server=optimizer_Server,
                scheduler_server=scheduler_Server,
                train_loader=USL_train_loader,
                test_loader=USL_test_loader,
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

    return net_glob_head, model_server, net_glob_tail


def warmup_evaluate(net_glob_head, model_server, net_glob_tail, test_loader, args):
    pass



def warmup_optimizer_step(
    _lm_loss,
    _attack_loss,
    head_hidden_states_for_tail,
    head_hidden_states_for_attack,
    hidden_states0,
    optimizer_head,
    optimizer_tail,
    ):
    _lm_loss.backward()
    dfx_1 = head_hidden_states_for_tail.grad.clone().detach()

    optimizer_tail.step()
    optimizer_tail.zero_grad()

    _attack_loss.backward()
    dfx_2 = head_hidden_states_for_attack.grad.clone().detach()
    grad_intermediate = dfx_1 - dfx_2
    hidden_states0.backward(grad_intermediate)

    optimizer_head.step()
    optimizer_head.zero_grad()

    
def warmup(
    model_head,
    model_tail,
    optimizer_head,
    optimizer_tail,
    train_loader,
    args,
    train_step,
    attack_model,
    epoch,
):
    model_head.train()
    model_tail.train()
    log_start_time = time.time()
    attack_model.train()
    print("warmup training...")
    device = "cuda"

    _lm_losses=[]
    _attack_losses=[]
    log_list = []

    for idx, data in enumerate(train_loader):
        # The client interacts with the server in turn
        _input = data["input_ids"].to(device)
        _target = data["input_ids"]
        _msk = data["attention_mask"]

        train_step += 1

        hidden_states0, head_presents, w_head = model_head(_input)
        head_hidden_states_for_tail = hidden_states0.clone().detach().requires_grad_(True)
        head_hidden_states_for_attack = hidden_states0.clone().detach().requires_grad_(True)
        _lm_logits, _lm_loss, w_tail = model_tail(
                _input.shape,
                head_hidden_states_for_tail,
                head_presents,
                lm_labels=_target,
                lm_mask=_msk,
                label_smooth=args.label_smooth,
            )
        logits = attack_model(head_hidden_states_for_attack)
        _attack_loss = calc_unshift_loss(logits, _input)

        _lm_losses.append(_lm_loss.item())
        _attack_losses.append(_attack_loss.item())

        warmup_optimizer_step(
                _lm_loss = _lm_loss,
                _attack_loss = _attack_loss,
                head_hidden_states_for_tail=head_hidden_states_for_tail,
                head_hidden_states_for_attack=head_hidden_states_for_attack,
                hidden_states0 = hidden_states0,
                optimizer_head = optimizer_head,
                optimizer_tail = optimizer_tail,
            )
       
        if train_step % args.warmup_log_interval == 0:
            elapsed = time.time() - log_start_time
            lr = optimizer_head.param_groups[0]["lr"]
            log_str = (
                f"| epoch: {epoch} step: {train_step} |"
                f"lr: {lr} | ms/batch {elapsed * 1000 / args.warmup_log_interval} |"
                f"lm_loss: {sum(_lm_losses) / len(_lm_losses)} | attack_loss: {sum(_attack_losses) / len(_attack_losses)} | "
            )

            log_list.append(log_str)

            if args.rank == 0:
                print(log_str)
            log_start_time = time.time()
            _lm_losses = []
            _attack_losses = []
        
        
    return train_step,model_head, model_tail

def warmup_training(lm_net_Head, lm_net_Tail, args, config, dataset, attack_model):
    
    tokenizer = AutoTokenizer.from_pretrained("/home/lzh/projects/privacy-split-llm/data/models/gpt2-large", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    warmup_dataset = get_dataset(dataset, tokenizer=tokenizer, client_ids=[])
    warmup_train_loader = warmup_dataset.get_dataloader_unsliced(4, 'train')

    optimizer_Head = create_adam_optimizer_from_args(lm_net_Head, args)
    optimizer_Tail = create_adam_optimizer_from_args(lm_net_Tail, args)
      
    # scheduler_Head = create_optimizer_scheduler(optimizer_Head, args)
    # scheduler_Tail = create_optimizer_scheduler(optimizer_Tail, args)

    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step, lm_net_Head, lm_net_Tail = warmup(
                model_head=lm_net_Head,
                model_tail=lm_net_Tail,
                optimizer_head=optimizer_Head,
                optimizer_tail=optimizer_Tail,
                train_loader=warmup_train_loader,
                args=args,
                train_step=train_step,
                attack_model=attack_model,
                epoch = epoch,
            )

            if train_step >= args.warmup_max_step :
                if args.rank == 0:
                    print("-" * 100)
                    print("End of training")
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print("-" * 100)
            print("Exiting from training early")

    return lm_net_Head, lm_net_Tail
