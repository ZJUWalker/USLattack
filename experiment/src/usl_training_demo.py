import sys
import os
import argparse
sys.path.append(os.path.abspath("/home/lzh/projects/privacy-split-llm"))
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
from sfl.utils.exp import get_model_and_tokenizer, get_fl_config, get_dataset, \
    add_sfl_params, args_to_dict, print_args
from data_utils import FT_Dataset
# from splitmodel import GPT2Config, GPT2LMModel_Server, GPT2LMModel_Client
from U_shaped_splitmodel import GPT2Config, GPT2LMModel_Head, GPT2LMModel_Server, GPT2LMModel_Tail
from exp_utils import create_exp_dir, print_trainable_parameters, inspect_model_parameters
from options import args_parser
import loralib as lora
from transformers import AutoTokenizer
from exp_utils import (
    create_exp_dir, 
    print_trainable_parameters, 
    inspect_model_parameters, 
    print_args, 
    AverageMeter, 
    fed_avg, 
    optimizer_step, 
    evaluate,
    train_validate)

torch.set_printoptions(threshold=100000) # set the print options for torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set the visible GPU device

args = args_parser()
parse_gpu(args)
print_args(args)

torch.manual_seed(args.random_seed)
random.seed(args.random_seed)

if args.rank == 0: # create logging directory
    args.logging = create_exp_dir(args.work_dir)


tokenizer = AutoTokenizer.from_pretrained("/home/lzh/projects/privacy-split-llm/data/models/gpt2-large", trust_remote_code=True)
dataset = get_dataset('wikitext', tokenizer=tokenizer, client_ids=[])
train_loader = dataset.get_dataloader_unsliced(args.train_batch_size, 'train')
valid_loader = dataset.get_dataloader_unsliced(args.valid_batch_size, 'validation')

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

lm_net_Head = GPT2LMModel_Head(config).cuda()
lm_net_Server = GPT2LMModel_Server(config).cuda()
lm_net_Tail = GPT2LMModel_Tail(config).cuda()

state_dict = torch.load(args.init_checkpoint)

if args.init_checkpoint is not None: # load model weight
    lm_net_Head.load_weight(state_dict)
    lm_net_Server.load_weight(state_dict)
    lm_net_Tail.load_weight(state_dict)
    print("loaded model pretrained weight.")

if args.lora_dim > 0: # mark lora-values as trainable
    print("marking lora-values as trainable.")
    lora.mark_only_lora_as_trainable(lm_net_Head)
    lora.mark_only_lora_as_trainable(lm_net_Server)
    lora.mark_only_lora_as_trainable(lm_net_Tail)

print_trainable_parameters(lm_net_Head, lm_net_Server, lm_net_Tail)

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
        args.max_epoch * len(train_loader) + args.world_size - 1
    ) // args.world_size
    # args.max_step = 1000
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
            train_loader=train_loader,
            test_loader=valid_loader,
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
