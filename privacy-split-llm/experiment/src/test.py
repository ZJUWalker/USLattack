import sys
import os
import argparse

sys.path.append(os.path.abspath("/home/lzh/projects/privacy-split-llm"))
from sfl.utils.exp import get_model_and_tokenizer, get_fl_config, get_dataset, \
    add_sfl_params, args_to_dict, print_args
from sfl.model.attacker.eia.eia_attacker import EmbeddingInversionAttacker
from sfl.model.attacker.sma_attacker import SmashedDataMatchingAttacker
from sfl.strategies.sl_strategy_with_attacker import SLStrategyWithAttacker
from sfl.utils.model import set_random_seed
from sfl.simulator.simulator import SFLSimulator, config_dim_reducer
from sfl.model.attacker.sip.sip_attacker import SIPAttacker
from sfl.model.attacker.dlg_attacker import TAGAttacker, LAMPAttacker
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/home/lzh/projects/SplitLLM/data/models/Llama-2-7b-chat-hf", trust_remote_code=True)
print(tokenizer.vocab_size)
dataset = get_dataset('wikitext', tokenizer=tokenizer, client_ids=[])
print(dataset)
data_loader = dataset.get_dataloader_unsliced(4)
print(len(data_loader))
print(data_loader)

for step, batch in enumerate(data_loader):
    # optimizer.zero_grad()
    input_ids = batch['input_ids'].to('cuda')
    attention_mask = batch['attention_mask'].to('cuda')
    labels = batch['labels'].to('cuda')
    # print(tokenizer.decode(input_ids[0]))
    # # print(attention_mask)
    # print(tokenizer.decode(labels[0]))

    print(input_ids[0])
    print(attention_mask[0])
    print(labels[0])
    print(batch)
    break