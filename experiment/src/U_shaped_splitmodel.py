#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2024, FDU
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the FDU nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict 
import copy
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parameter import Parameter

import loralib as lora


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = lora.MergedLinear(
            nx, n_state * 3, 
            r=config.lora_attn_dim, 
            lora_alpha=config.lora_attn_alpha, 
            lora_dropout=config.lora_dropout, 
            enable_lora=[True, False, True], 
            fan_in_fan_out=True,
            merge_weights=False
        )
        self.c_proj = Conv1D(n_state, nx)

        self.config = config
    
    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10) 

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq_length, head_features)

    def forward(self, x, history=None, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        #_input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class GPT2Config(object):
    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        lora_attn_dim=0,
        lora_attn_alpha=128,
        lora_dropout=0.0,
        lora_r_dropout=0.0,
        fix_dropout=0.0,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.lora_attn_dim = lora_attn_dim
        self.lora_attn_alpha = lora_attn_alpha
        self.lora_dropout = lora_dropout
        self.lora_r_dropout = lora_r_dropout

        self.fix_dropout = fix_dropout


class Block(nn.Module): # Transformer block
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model_Head(nn.Module):
    def __init__(self, config):
        super(GPT2Model_Head, self).__init__()
        self.n_layer = config.n_layer # layer number
        self.n_embd = config.n_embd # embedding dimension
        self.n_vocab = config.vocab_size # vocabulary size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # word embedding
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # position embedding
        block = Block(config.n_ctx, config, scale=True) # transformer block
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(3)])

        self.config = config

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        past=None,
        len_past=None
    ):
        
        if past is None: # past is None or empty list
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None: # past is not None but len_past is None
            # equal size for past. []
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:  # get position embeddings
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length,
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1) #.long()

        input_shape = input_ids.size()
    
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None: # if token type are not None, add token type embeddings
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds # get input hidden states
        presents = []

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states, present = block(hidden_states, layer_past=layer_past, len_past=len_past)
            presents.append(present)

        return hidden_states, presents


class GPT2Model_Tail(nn.Module):
    def __init__(self, config):
        super(GPT2Model_Tail, self).__init__()
        self.n_layer = config.n_layer # layer number
        self.n_embd = config.n_embd # embedding dimension
        self.n_vocab = config.vocab_size # vocabulary size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd) 
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(3)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config

    def forward(
        self,
        hidden_states,
        presents,
        input_shape,
        past=None,
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            past_length = past[0][0].size(-2)

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states, present = block(hidden_states, layer_past=layer_past, len_past=len_past)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents


class GPT2Model_Server(nn.Module):
    def __init__(self, config):
        super(GPT2Model_Server, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        # self.wte = nn.Embedding(config.vocab_size, config.n_embd) # 
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer-6)])
        # self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config

    def forward(
        self,
        hidden_states,
        presents,
        input_shape,
        past=None,
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            past_length = past[0][0].size(-2)

        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            hidden_states, present = block(hidden_states, layer_past=layer_past, len_past=len_past)
            presents.append(present)

        # hidden_states = self.ln_f(hidden_states)
        # output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states, presents


class GPT2LMHead(nn.Module): # Language Modeling Head
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2LMModel_Head(nn.Module):
    def __init__(self, config):
        super(GPT2LMModel_Head, self).__init__()
        self.transformer_Head = GPT2Model_Head(config)

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer_Head.wte.weight)

    def forward(
            self,
            input_ids,
            past=None,
            len_past=None,
    ):
        _batch, _len = input_ids.shape
        hidden_states_head, presents_head = self.transformer_Head(input_ids, past=past,len_past=len_past)
        
        return hidden_states_head, presents_head, self.transformer_Head.state_dict()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"

            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        for n, p in self.transformer_Head.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        self.transformer_Head.load_state_dict(state_dict, strict=False)


class GPT2LMModel_Server(nn.Module):
    def __init__(self, config):
        super(GPT2LMModel_Server, self).__init__()

        self.transformer_Server = GPT2Model_Server(config)
        # self.lm_head = GPT2LMHead(self.transformer_Server.wte.weight, config)
        self.apply(self._init_weights)

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer_Server.wte.weight)

    def forward(
            self,
            input_ids_shape,
            hidden_states_head,
            presents_head,
    ):
        _batch, _len = input_ids_shape

        hidden_states_server, presents_server = self.transformer_Server(hidden_states_head,
                                                                                    presents_head, input_ids_shape)
        # return lm_logits, presents_server
        return hidden_states_server, presents_server, self.transformer_Server.state_dict()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []
        for key in state_dict_tmp:
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"

            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        for n, p in self.transformer_Server.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('h.'):
                parts = key.split('.')
                layer_idx = int(parts[1])
                new_key = '.'.join(['h', str(layer_idx - 3)] + parts[2:])
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        self.transformer_Server.load_state_dict(new_state_dict, strict=False)


class GPT2LMModel_Tail(nn.Module):
    def __init__(self, config):
        super(GPT2LMModel_Tail, self).__init__()

        self.transformer_Tail = GPT2Model_Tail(config)
        self.lm_head = GPT2LMHead(self.transformer_Tail.wte.weight, config)
        self.apply(self._init_weights)
        self.task_type = 'ALM' 

    def set_tied(self):
        """ Make sure we are sharing the embeddings"""
        self.lm_head.set_embeddings_weights(self.transformer_Tail.wte.weight)

    def forward(
            self,
            input_ids_shape,
            hidden_states_server,
            presents_server,
            lm_labels=None,
            lm_mask=None,
            label_smooth=0.0,
            is_report_accuracy=False,
            vocab_size=0,
    ):
        _batch, _len = input_ids_shape

        hidden_states_tail, presents_tail = self.transformer_Tail(hidden_states_server, presents_server, input_ids_shape)

        # batch, seq, vocab
        lm_logits = self.lm_head(hidden_states_tail)

        if lm_labels is not None: 
            if is_report_accuracy: # report accuracy
                _pred_token = torch.argmax(lm_logits, dim=-1)
                _hit = (_pred_token == lm_labels) * lm_mask

                _t1_acc = torch.zeros(_batch, dtype=torch.float)
                _all_acc = torch.zeros(_batch, dtype=torch.float)

                for _b in range(0, _batch):
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] > 0:
                                _t1_acc[_b] = 1.0
                            break

                    _is_succ = True
                    for _i in range(0, _len):
                        if lm_mask[_b, _i] >= 1.0:
                            if _hit[_b, _i] <= 0:
                                _is_succ = False
                                break

                    if _is_succ:
                        _all_acc[_b] = 1.0

                # _t1_acc = _t1_acc * 1.0 / _batch
                # _all_acc = _all_acc * 1.0 / _batch
            if self.task_type != 'ALM':
                if label_smooth > 0.0001: # get (label smooth) loss
                    logprobs = torch.nn.functional.log_softmax(lm_logits.view(-1, lm_logits.size(-1)), dim=-1)
                    nll_loss = -logprobs.gather(dim=-1, index=lm_labels.view(-1).unsqueeze(1))
                    nll_loss = nll_loss.squeeze(1)
                    smooth_loss = -logprobs.mean(dim=-1)
                    loss = (1.0 - label_smooth) * nll_loss + label_smooth * smooth_loss
                    loss = loss.view(_batch, _len)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduce=False)
                    print(lm_logits.shape, lm_labels.shape)
                    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(_batch, _len)

                if lm_mask is None:
                    lm_mask = torch.ones(loss.shape, dtype=loss.dtype, device=loss.device)
                loss = loss * lm_mask

                loss = loss.sum() / (lm_mask.sum() + 0.0001)
                loss = loss.mean()
            else:
                # 移位操作，去掉最后一个时间步和第一个时间步
                shift_logits = lm_logits[..., :-1, :].contiguous()  # 去掉最后一个时间步
                shift_labels = lm_labels[..., 1:].contiguous()     # 去掉第一个时间步

                # 检查形状是否匹配，确保总元素数可以正确 reshape
                batch_size, seq_len_minus_one, vocab_size = shift_logits.shape
                assert shift_labels.shape == (batch_size, seq_len_minus_one), "shift_logits and shift_labels must align."

                shift_logits = shift_logits.view(-1, vocab_size)  # (batch_size * (seq_len-1), vocab_size)
                shift_labels = shift_labels.view(-1)             # (batch_size   * (seq_len-1))
                shift_labels = shift_labels.to(shift_logits.device)

                # 定义交叉熵损失函数并计算损失
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)        
            if is_report_accuracy:
                return lm_logits, loss, _t1_acc, _all_acc
            else:
                return lm_logits, loss, self.transformer_Tail.state_dict()
        return lm_logits, presents_tail, self.transformer_Tail.state_dict()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def load_weight(self, state_dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        state_dict_tmp = copy.deepcopy(state_dict)
        old_keys = []
        new_keys = []

        for key in state_dict_tmp: # 遍历 state_dict 中的所有键，进行键名调整
            new_key = None
            if key.endswith(".g"):
                new_key = key[:-2] + ".weight"
            elif key.endswith(".b"):
                new_key = key[:-2] + ".bias"
            elif key.endswith(".w"):
                new_key = key[:-2] + ".weight"

            if key.startswith("module.transformer."):
                new_key = key[len("module.transformer."):]

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        for n, p in self.transformer_Tail.named_parameters():
            if n not in state_dict:
                state_dict[n] = p

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('h.'):
                parts = key.split('.')
                layer_idx = int(parts[1])
                new_key = '.'.join(['h', str(layer_idx - 21)] + parts[2:])
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        self.transformer_Tail.load_state_dict(new_state_dict, strict=False)
        self.set_tied()