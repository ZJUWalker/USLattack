import sys
import os
sys.path.append(os.path.abspath("/home/lzh/projects/privacy-split-llm"))
import os
import torch
import random
import pandas as pd
from gpu import add_gpu_params, parse_gpu, distributed_opt, distributed_sync, cleanup
from usl_train import usl_train, warmup_training
from sfl.utils.exp import get_dataset, print_args

# from sfl.model.attacker.sip.inversion_training import evaluate_rouge
from U_shaped_splitmodel import GPT2Config, GPT2LMModel_Head, GPT2LMModel_Server, GPT2LMModel_Tail
from exp_utils import (
    create_exp_dir, 
    create_exp_dir, 
    print_args,
    SIPAttackTraining,
    SIPAttackEvaluate)
from options import args_parser, SIP_train_args_parser
from transformers import AutoTokenizer

torch.set_printoptions(threshold=100000) # set the print options for torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set the visible GPU device

if __name__ == "__main__":
    args = args_parser()
    SIP_train_args = SIP_train_args_parser()
    parse_gpu(args)
    print_args(args)

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    if args.rank == 0: # create logging directory
        args.logging = create_exp_dir(args.work_dir)

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
    config.is_ALM = True if args.is_ALM == True else False
    
    lm_net_Head = GPT2LMModel_Head(config).cuda()
    lm_net_Server = GPT2LMModel_Server(config).cuda()
    lm_net_Tail = GPT2LMModel_Tail(config).cuda()

    state_dict = torch.load(args.init_checkpoint)

    if args.init_checkpoint is not None: # load model weight
        lm_net_Head.load_weight(state_dict)
        lm_net_Server.load_weight(state_dict)
        lm_net_Tail.load_weight(state_dict)
        print("loaded model pretrained weight.")

    print("="*50, "SIPAttackTraining", "="*50)
    attack_model = SIPAttackTraining(config, SIP_train_args, lm_net_Head)
    print("="*48, "SIPAttackTraining_End", "="*48)

    print("Evaluate attack model on test set...")
    SIPAttackEvaluate(SIP_train_args, attack_model, lm_net_Head)

    print('='*48, "warmup training...", '='*48)
    dataset = "imdb"
    lm_net_Head, lm_net_Tail = warmup_training(lm_net_Head, lm_net_Tail, args, config, dataset, attack_model)
    print('='*44, "warmup training end...", '='*44)

    print('='*44, "SIPAttack after warmup training", '='*44)
    SIPAttackEvaluate(SIP_train_args, attack_model, lm_net_Head)

    print('='*44, "U-shaped split model training...", '='*44)
    net_glob_head, model_server, net_glob_tail = usl_train(args, config, state_dict, lm_net_Head, lm_net_Server, lm_net_Tail, dataset)

    print("Evaluate attack model on trained model on test set...")
    SIPAttackEvaluate(SIP_train_args, attack_model, net_glob_head, valid_dataset = dataset)