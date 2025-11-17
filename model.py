"""This is my version of the GPT model."""


"""Summary of System

GPT is made from main peices
1) Embeddings. Tokens -> Vectors
2) Transformer Blocks
    - LayerNorm
    - Multi-head Causual Self Attention
    - Multi layer perceptron feed forward network

3) Stack of Blocks (deep transformer)
4) Final output layer (LM head)
5) Training Utilites (optimizer, loss)
6 Generation loop (Auto regressive text generation)


"""
import math
imp;otr inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """Layer normalization module but with optional bias"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(troch.ones(ndim))
        self.bias = nn.Parameter(torch.zeroes(ndim))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        #Now we are going to build the key, query, and value projectsiona for all heads, but not in a batcvh
        self.c_attn == nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)

        #output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        #regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.dropout = config.dropout

        #flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("Warning using slow attention. flash attention needs higher pytorch")

            #casual self attention
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
    
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        #causal self-attention, Self - attend (B, nh, T, hs)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_casual=true)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -2) * (1.0 / math.sqrt(k.size(-1))))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contigous().view(B, T, C)

        #output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Linear Expansion = 4 x Embedding Size
    Note) We use GELU Activation, over the RELU Activation, so we can use OAI Checkpoints
    Linear Redfuctioun -> Embedding size

    Why? 
        After attention mixes information across positions, MLPs transform that information non linearly
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias= config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.Dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    1) LayerNorm
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm()