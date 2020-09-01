"""
# https://github.com/bentrevett/pytorch-seq2seq/issues/129
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
"""
import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
from IPython.core.debugger import set_trace #set_trace()

from layer_norm import  *


######## ENCODER PART #################################

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 512):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        # self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, src len]
        
        batch_size = src.shape[0]
        # batch_size = 1
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        src = torch.tensor(src).to(self.device).long()
        src = src.cuda()
        self.scale = self.scale.cuda()
        src = self.dropout((self.tok_embedding(src) * self.scale)) #  + self.pos_embedding(pos))
       
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

####        ATTENTION LAYER  #################################################################

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):  
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(self.head_dim, self.head_dim)
        self.fc_k = nn.Linear(self.head_dim, self.head_dim)
        self.fc_v = nn.Linear(self.head_dim, self.head_dim)
        
        self.fc_o = nn.Linear(self.n_heads * self.head_dim, self.hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        # batch_size = 1

        # split embedding into self. head pieces
        values = value.reshape(N, value_len, self.n_heads, self.head_dim)
        keys = key.reshape(N, key_len, self.n_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.n_heads, self.head_dim)
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(queries)
        K = self.fc_k(keys)
        V = self.fc_v(values)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        '''        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        print("**********************************")
        
        print(f' Shape of Q is {Q.shape}')
        print("**********************************")
        print(f' Shape of K is {K.shape}')
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        import pdb;pdb.set_trace
        print("**********************************")
        print(f' Shape of mask is {mask.shape}')
        print("**********************************")
        print(f' Shape of Energy is {energy.shape}')
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        '''
        print(f' Shape of Q is {Q.shape}')
        print("**********************************")
        print(f' Shape of K is {K.shape}')
        
        Q = Q.cuda()
        K = K.cuda()
        self.scale =self.scale.cuda()
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        # import pdb;pdb.set_trace
        print("**********************************")
        print(f' Shape of mask is {mask.shape}')
        print("**********************************")


        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        # queries shape : (N, query_len, heads, heads_dim)
        # keyshape shape : (N, key_len, heads, heads_dim)
        # energy shape : (N, heads, query_len, key_len)

        if mask is not None:
            energy =  energy.masked_fill(mask == 0, float("-1e10")) # for numerical stability
        
        attention = torch.softmax(energy / (self.hid_dim ** (1/2)), dim=3) # Attention(Q,K,V) = sofmax(QK^{T}/(d_{k})**(1/2)) * V


        # print("**********************************")
        # print(f' Shape of batch is {batch_size}')
        # print("**********************************")
        # print(f' Shape of batch is {query_len}')
        # print(f' Shape of batch is {query_len}')

        out = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            N, query_len, self.n_heads * self.head_dim 
        )

        x = self.fc_o(out)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

###### decoder part ###############

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 512):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        # self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
                
        # batch_size = trg.shape[0]
        batch_size = 1
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
        trg = torch.tensor(trg).to(self.device).long()     
        trg = trg.cuda()
        self.scale = self.scale.cuda()   
        trg = self.dropout((self.tok_embedding(trg) * self.scale)) #  + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention


############# decoder Layer  #################################

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        # Frame level importance score regression
        # Two layer NN 
        self.m = 256  # TODO Need to change this as a common parameter
        self.ka = nn.Linear(in_features=self.m, out_features=256)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=256)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=256)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        
    def make_src_mask(self, src):
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):

        m = src.shape[1] # Feature size
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)

        x = src.view(-1,m)
        
        #enc_src = [batch size, src len, hid dim]
                
        y, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        y = y # + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        
        return y, attention


# if __name__ == "__main__":
#     set_trace()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
#         device
#     )
#     trg =  torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

#     src_pad_idx = 0
#     trg_pad_idx = 0
#     src_vocab_size = 10
#     trg_vocab_size = 10
#     model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
#         device
#     )
#     out, attention = model(x, trg[:, :-1])
#     print(out.shape)


# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)

# if __name__ == "__main__":
#     pass

'''
if __name__ == "__main__":
    # set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = 1024
    OUTPUT_DIM = 1024
    HID_DIM = 256 
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 1024 # try this also with 1024
    DEC_PF_DIM = 1024
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    SRC_PAD_IDX = 0
    TRG_PAD_IDX = 0

    enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                device)

    dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                device)

    # src_vocab_size = 1024
    # trg_vocab_size = 1024

    # x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2, 0]]).to(
    #     device, dtype=torch.int64
    # )
    # trg =  torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0,0,0], [1, 5, 6, 2, 4, 7, 6, 2,0,0]]).to(device, dtype=torch.int64)

    x = torch.tensor([[0.1,0.8,0.8,0.4,0.9,0.4,0.4,0.5,0.4,0.2,0.3,0.8,0.2,0.7,0.6,0.1,0.2,0.1,0.1,0.4,0.2,0.1,0.0,0.5,0.2,0.4,0.3,0.3,0.7,0.1,0.4,0.6,0.5,1.0,0.1,0.8,0.9,0.0,0.2,0.9,0.8,0.0,0.9,0.7,0.2,0.2,0.9,0.6,0.1,0.2,0.6,0.0,0.1,0.1,0.3,0.5,0.8,0.8,0.4,0.4,0.7,0.7,0.4,0.2,0.1,1.0,0.3,0.8,0.1,0.7,0.7,0.9,0.6,0.3,0.8,0.2,0.9,0.6,0.7,0.8,0.2,0.1,1.0,0.6,0.5,0.5,0.5,0.8,0.8,0.3,0.1,0.2,0.5,0.9,0.6,0.8,0.0,0.6,0.2,0.1,0.8,0.4,0.8,0.5,0.8,0.4,0.7,0.6,0.8,0.1,0.4,0.8,1.0,0.9,0.4,0.4,0.4,0.1,0.7,0.3,0.8,0.6,0.4,0.5,0.9,0.1,0.9,0.7,0.4,0.7,0.1,0.8,0.2,0.2,0.7,0.2,0.9,0.6,0.2,0.9,0.1,0.9,0.2,1.0,0.9,0.6,0.3,0.6,0.9,0.6,0.0,0.3,0.4,0.6,0.7,0.9,0.2,0.6,0.2,0.5,0.3,0.3,0.4,0.4,0.1,0.2,0.6,0.0,0.7,0.5,0.5,0.2,0.5,0.6,0.5,0.5,0.7,0.8,0.4,0.5,0.8,0.8,0.1,0.5,0.7,0.8,0.1,0.1,0.8,0.6,0.6,0.4,1.0,0.4,0.6,0.9,0.1,0.6,0.3,1.0,0.7,0.2,0.5,0.5,1.0,0.5,0.4,0.3,0.7,0.1,1.0,0.9,0.4,0.6,0.6,0.6,0.2,0.0,0.9,0.9,0.2,0.1,0.5,0.5,0.8,0.7,0.8,0.0,0.0,0.1,0.5,0.5,0.5,0.8,0.1,0.5,1.0,0.3,0.2,0.8,0.9,0.4,0.4,0.9,0.2,0.4,0.9,0.9,0.3,0.7,0.4,0.9,0.5,0.7,0.8,0.5,0.5,0.5,0.8,0.7,0.9,0.2,0.8,1.0,0.1,0.9,0.6,0.5,0.0,0.2,0.8,0.2,0.8,0.5,0.9,0.9,0.5,0.6,0.1,0.8,1.0,0.3,0.1,0.5,0.9,0.1,0.0,0.5,0.3,0.1,0.5,0.8,0.3,0.4,0.4,0.3,0.2,0.8,0.7,0.6,0.3,0.5,0.1,0.7,0.4,0.2,0.1,0.1,0.4,0.2,0.8,0.8,0.4,0.1,0.0,0.3,0.2,0.0,1.0,0.2,0.6,0.5,0.7,0.7,0.7,0.1,0.2,0.1,0.1,0.9,0.6,0.5,1.0,0.4,0.4,0.8,0.7,0.5,0.6,0.9,0.0,0.8,0.3,0.1,0.5,0.9,0.9,0.9,0.7,0.7,1.0,0.6,0.6,1.0,0.8,1.0,0.4,0.3,0.2,1.0,0.9,0.2,0.7,0.1,0.3,0.1,0.1,0.7,0.6,0.8,0.8,0.7,0.7,0.4,0.8,0.4,0.1,0.0,1.0,0.2,0.6,0.8,0.3,0.9,0.3,0.6,0.6,0.4,0.7,0.0,0.2,0.9,0.2,0.1,0.4,0.9,0.5,0.2,0.4,1.0,0.1,0.3,0.8,0.8,0.2,0.2,0.6,0.8,0.1,0.0,0.5,1.0,0.5,0.7,0.3,0.5,0.0,0.2,0.6,0.7,0.6,0.4,0.2,0.0,0.4,0.4,0.0,0.3,0.3,0.8,0.5,0.7,0.4,0.1,0.8,0.4,0.1,0.3,1.0,0.3,0.6,0.5,0.6,0.2,0.9,0.4,0.4,0.8,0.0,0.3,0.8,0.3,0.1,0.0,0.5,0.5,0.8,0.6,1.0,0.7,0.8,0.7,0.7,0.6,0.0,0.6,0.6,0.3,0.7,0.2,1.0,0.6,0.4,0.8,0.4,0.7,0.3,0.8,0.8,0.1,0.1,0.2,0.2,0.7,0.1,0.8,0.4,1.0,0.6,1.0,0.3,0.9,0.9,0.9,0.9,1.0,0.2,0.3,0.9,0.5,0.5,0.4,0.1,0.4,0.0,0.7,0.2,0.6,0.8,0.2,0.8,0.2,0.6,0.9,0.1,0.3,0.4,0.2,0.9,0.3,0.9,0.1,0.1,0.7,1.0,0.4,0.2,0.9,0.2,0.5,0.1,0.3,0.6,0.5,0.6,0.5,0.3,0.4,0.3,0.9,0.7,0.1,0.2,0.8,1.0,0.5,0.0,0.8,0.2,0.2,0.0,1.0,0.2,1.0,0.5,1.0,0.9,0.5,0.2,0.5,0.8,0.4,0.9,0.9,0.2,0.5,0.5,0.2,0.6,0.3,0.3,0.8,0.3,0.5,0.4,0.2,0.7,0.8,0.9,0.2,0.9,0.6,0.0,0.3,0.8,0.5,0.3,0.9,0.9,0.7,0.4,0.9,0.3,0.7,0.4,0.3,0.5,0.8,0.9,0.7,0.6,0.5,0.1,0.9,0.6,0.5,0.2,0.7,0.3,0.3,0.1,0.0,0.2,0.5,0.9,0.7,0.3,0.3,1.0,0.3,0.6,0.9,0.1,0.9,0.3,0.7,0.1,0.7,0.6,0.6,0.5,0.1,0.1,0.3,0.5,0.7,0.1,0.7,0.4,0.8,0.4,0.6,0.8,0.7,0.6,0.0,0.1,0.3,0.8,0.2,0.5,0.7,0.0,0.4,1.0,0.2,0.2,0.4,0.3,0.9,0.2,0.4,0.3,0.4,0.2,0.5,0.6,0.6,0.8,0.7,0.3,0.1,0.7,0.5,0.1,0.4,1.0,0.2,0.8,0.5,0.7,0.3,0.7,0.6,0.7,0.5,1.0,0.2,0.8,0.0,0.1,0.2,0.6,0.0,0.2,0.1,0.2,0.4,0.6,0.2,1.0,0.3,0.1,0.1,0.7,0.0,0.7,0.0,0.7,0.9,0.1,0.2,0.8,0.7,0.5,0.3,0.8,0.3,0.0,0.1,0.1,0.8,0.9,0.2,0.5,0.5,0.4,0.4,0.8,0.9,0.4,1.0,0.8,0.4,0.2,0.1,0.3,0.1,0.7,0.9,0.2,0.9,0.8,0.7,0.2,0.7,0.4,0.0,1.0,0.7,0.3,0.6,0.9,0.1,0.5,0.2,0.5,0.7,0.3,0.9,0.7,0.2,1.0,0.6,0.4,0.3,0.1,0.1,0.0,0.3,0.9,0.7,0.5,0.9,0.8,0.6,0.8,0.1,0.4,0.5,0.8,0.7,0.4,0.8,0.4,0.1,0.6,0.8,0.0,0.9,0.7,0.7,0.7,0.7,0.3,0.4,0.4,0.2,0.6,0.3,0.4,1.0,0.2,0.3,0.0,0.5,1.0,0.8,0.7,0.3,0.2,0.7,0.1,0.5,0.2,0.3,0.4,0.8,0.4,0.2,0.3,0.9,0.5,0.1,0.7,0.0,0.3,0.3,0.1,0.1,0.8,0.2,0.6,0.2,0.0,0.3,0.6,0.4,0.7,0.6,0.2,0.8,0.4,0.3,0.7,0.3,0.7,0.9,0.4,0.8,0.9,0.4,0.5,0.4,0.6,0.7,0.5,0.6,0.6,0.4,0.4,0.8,0.3,0.9,0.8,0.9,0.6,0.1,0.9,1.0,1.0,0.8,0.8,0.2,0.1,0.1,0.4,0.9,0.9,0.9,0.6,0.4,0.8,0.6,0.6,0.4,0.6,0.6,0.8,1.0,0.2,0.3,0.4,0.9,0.3,0.7,0.9,0.6,1.0,0.5,0.3,0.5,0.9,0.1,0.9,0.6,0.4,0.9,0.9,0.7,0.9,0.0,0.3,0.7,0.2,0.1,0.2,0.6,0.1,0.6,0.3,0.5,0.1,0.5,0.7,0.1,0.9,0.4,0.1,0.4,1.0,0.1,0.7,0.5,0.6,0.1,0.4,1.0,0.3,0.8,0.3,0.9,0.8,0.9,0.4,0.2,0.2,0.7,0.0,0.8,0.7,0.3,0.2,0.2,0.3,0.9,0.8,0.2,0.3,0.4,0.2,0.9,0.4,0.6,0.2,0.5,0.6,0.0,0.3,0.2,0.9,0.7,0.5,0.7,0.8,0.8,0.2,0.7,0.7,0.5,0.1,0.0,0.3,0.6,0.4,1.0,1.0,0.1,0.2,0.4,0.5,0.0,0.2,0.6,0.8,0.7,0.5,0.2,0.3,0.7,0.4,0.7,0.8,0.2,0.7,0.8,0.9,0.7,0.2,0.5,0.7,0.9,0.7,0.5,0.1,1.0,0.5,0.6,0.9,0.5,0.7,0.3,0.9,0.8],
    [0.5,0.6,0.0,0.9,0.9,0.4,0.4,0.9,0.1,0.7,0.8,0.7,1.0,0.5,0.6,0.5,0.9,0.7,0.2,0.4,0.6,0.7,0.4,0.2,0.3,0.3,0.9,1.0,0.0,0.5,0.5,0.6,0.1,0.6,0.1,1.0,0.8,0.4,0.2,0.6,0.9,0.2,0.1,0.5,0.0,0.5,0.3,0.9,0.5,0.0,0.9,0.4,0.4,0.5,0.7,0.9,0.1,0.9,0.0,0.2,0.6,0.8,0.7,0.1,0.6,0.2,0.2,0.8,0.7,0.2,0.1,0.2,0.6,0.8,0.6,0.4,0.8,0.8,0.9,0.7,0.8,0.4,0.5,0.1,0.7,0.9,0.2,0.3,0.0,0.7,0.0,0.1,0.7,0.8,0.9,0.7,0.6,0.3,0.7,0.7,0.2,0.1,0.3,0.7,0.3,0.8,0.2,0.1,0.8,0.9,0.2,0.4,0.5,0.5,0.9,0.9,0.3,0.7,0.1,0.6,0.7,0.2,0.6,0.9,0.8,0.7,0.0,0.4,0.1,0.6,0.5,0.1,0.8,0.7,0.9,0.7,0.5,0.7,0.8,0.8,0.2,0.5,0.3,0.4,0.8,0.4,0.1,0.3,0.4,0.3,0.4,0.7,0.4,0.7,0.9,0.2,0.8,0.3,0.8,0.3,0.8,0.7,0.3,0.4,0.4,0.6,0.1,0.3,0.6,0.5,0.9,0.7,0.3,0.6,0.5,0.3,0.4,0.2,0.8,0.3,0.1,0.9,0.9,0.6,0.1,0.4,0.2,0.4,0.8,0.9,0.1,0.4,0.8,0.5,0.4,0.8,0.9,1.0,0.1,0.8,0.8,0.8,0.8,0.8,0.3,0.1,1.0,0.2,0.9,0.2,0.9,0.7,0.9,1.0,0.4,0.2,0.5,0.4,0.3,0.2,0.1,0.1,0.8,0.7,0.0,0.3,1.0,1.0,0.0,0.5,0.0,0.5,0.6,0.8,0.2,0.4,0.0,0.8,0.5,0.8,0.6,0.3,0.4,0.7,0.9,0.0,0.8,0.7,0.9,0.9,0.2,0.3,0.3,0.9,0.3,0.3,0.3,0.6,0.8,0.5,0.5,0.0,0.5,0.8,1.0,0.4,1.0,0.3,0.5,0.5,0.6,0.6,0.7,0.1,0.3,0.6,0.4,0.2,0.8,1.0,0.6,0.9,0.7,0.5,0.1,0.7,0.6,1.0,0.4,0.9,0.3,0.6,0.1,1.0,0.8,0.7,0.7,0.5,0.0,0.6,0.5,1.0,0.6,0.9,0.8,0.9,0.7,1.0,0.9,1.0,0.3,0.2,0.5,0.3,0.8,0.1,0.9,0.6,0.9,0.9,0.3,0.4,0.1,0.6,0.0,0.0,0.2,0.2,0.9,0.9,0.6,1.0,0.2,0.7,1.0,0.8,1.0,0.2,0.3,0.3,0.9,0.5,0.1,0.2,0.5,0.9,0.1,0.5,0.2,1.0,0.7,0.4,0.2,0.1,0.4,0.4,0.7,0.8,0.3,0.6,0.0,1.0,0.8,1.0,0.1,0.2,0.9,0.4,0.8,0.0,0.0,1.0,0.1,0.3,0.0,0.7,0.6,0.9,0.4,0.4,0.9,0.4,0.8,0.7,0.7,0.5,0.3,0.6,0.5,0.5,0.5,0.9,0.8,0.4,0.8,0.6,0.4,0.2,0.9,1.0,0.8,0.2,0.2,0.8,0.9,0.7,0.1,0.8,0.7,0.3,0.1,0.2,0.3,0.6,0.6,0.6,0.7,0.4,0.1,0.9,0.5,0.5,0.5,0.4,0.6,0.2,0.7,0.6,0.3,0.3,0.2,0.4,0.2,0.9,0.9,0.9,0.7,0.8,0.3,0.0,0.4,0.1,0.9,0.6,0.3,0.0,0.7,0.1,0.8,0.6,0.3,0.6,0.8,0.2,0.1,0.4,0.8,0.9,1.0,0.7,0.8,0.1,0.4,0.1,0.4,0.9,0.4,0.6,0.7,0.2,0.5,0.6,0.8,0.6,0.6,0.9,0.7,0.4,0.3,0.5,0.1,0.8,0.9,0.4,0.0,0.4,0.0,0.3,0.6,0.8,0.1,0.4,0.1,0.6,0.7,0.1,0.0,0.0,0.0,0.8,0.7,0.6,0.8,0.6,0.9,0.1,0.4,0.0,0.4,0.0,0.4,0.7,0.5,0.1,0.9,0.3,0.3,0.1,0.3,0.6,0.6,0.8,0.8,0.9,0.2,0.0,0.6,0.3,1.0,0.6,0.7,1.0,1.0,0.9,0.4,0.1,0.6,0.9,0.1,0.1,0.1,0.2,0.5,0.0,0.8,0.5,0.0,0.8,0.4,0.1,0.2,0.2,0.8,0.9,0.6,0.3,0.2,0.5,0.0,0.1,0.1,0.8,0.9,1.0,0.8,0.2,0.8,0.3,0.8,0.2,0.0,0.1,1.0,0.7,0.1,0.8,0.2,0.5,0.3,0.6,0.1,0.7,0.7,0.5,0.2,0.3,0.5,0.5,1.0,0.2,0.3,0.4,0.1,0.1,0.7,1.0,0.7,0.6,0.9,1.0,0.4,0.8,0.1,0.4,0.1,0.9,0.7,0.4,0.0,0.0,0.3,0.3,0.5,0.6,0.3,0.8,0.5,0.3,0.1,0.9,0.5,0.1,0.3,0.9,0.4,0.3,0.4,0.2,0.9,0.5,0.4,0.9,0.8,0.9,0.9,0.9,0.6,0.6,0.3,0.4,0.3,0.3,0.4,0.4,0.2,0.3,0.7,0.1,0.4,0.1,0.7,0.2,0.7,0.7,0.1,0.3,1.0,0.4,0.4,0.0,0.1,0.4,0.6,0.9,0.5,0.1,0.6,0.9,0.1,0.2,0.4,0.5,0.5,0.1,0.7,0.0,0.1,1.0,0.6,0.1,0.5,0.7,0.2,0.7,0.1,0.1,0.5,0.5,0.2,0.7,0.0,0.9,0.3,0.2,0.9,0.2,0.2,0.5,0.5,0.6,0.3,0.4,0.9,0.4,0.5,0.8,0.1,0.4,0.5,0.9,0.5,0.4,0.3,1.0,0.7,0.5,0.1,0.0,0.3,0.0,0.5,0.5,0.9,0.6,0.3,0.7,0.1,0.9,0.1,0.9,0.1,0.8,0.0,0.9,0.0,0.0,0.7,0.6,1.0,0.5,0.9,0.7,0.4,0.5,0.6,0.3,0.6,0.9,0.4,0.3,0.3,1.0,0.2,1.0,0.3,0.7,0.9,0.8,0.8,0.7,0.6,0.6,0.8,0.5,0.3,0.4,0.5,0.1,0.3,0.4,0.0,0.2,0.8,0.3,1.0,0.5,0.0,0.7,0.9,0.3,0.3,0.9,0.9,0.5,0.0,0.0,0.6,0.7,0.6,0.5,0.1,0.8,0.3,0.3,0.1,0.7,0.0,0.6,0.0,0.1,0.9,0.1,0.4,0.1,0.5,1.0,0.3,0.2,0.8,0.6,0.3,0.5,0.3,0.1,0.9,0.1,0.9,0.9,0.1,0.8,0.7,0.8,0.3,0.5,1.0,0.1,0.7,0.4,0.7,0.7,0.9,0.9,1.0,0.3,0.8,0.3,0.3,0.5,0.2,0.6,0.4,0.5,0.7,0.8,0.9,0.8,0.9,0.2,0.0,0.5,0.2,1.0,0.7,0.4,0.1,0.6,0.6,0.0,0.4,0.6,0.6,0.4,0.1,0.7,1.0,0.1,0.4,0.3,0.9,0.1,0.0,0.1,0.6,0.1,1.0,0.1,0.3,0.3,0.4,0.3,0.8,0.2,0.5,0.1,0.3,0.8,0.7,0.0,0.4,0.5,0.2,0.0,0.5,0.8,0.2,0.6,0.9,0.8,0.9,0.5,0.7,0.5,0.9,0.9,0.3,0.5,0.3,1.0,0.8,0.7,0.9,0.6,0.6,0.5,0.8,0.2,0.7,0.6,0.3,0.1,0.9,0.2,0.4,0.9,0.3,0.2,0.5,0.5,0.9,0.2,1.0,0.9,0.8,0.2,0.2,1.0,0.4,0.4,0.6,0.8,0.3,0.2,0.6,0.0,0.5,0.9,0.6,0.3,0.4,0.8,0.5,0.6,0.7,0.6,0.0,0.1,0.3,0.7,0.4,0.1,0.2,0.7,0.2,0.3,0.8,0.2,0.4,0.2,1.0,1.0,0.7,0.8,0.2,0.5,0.3,0.5,0.4,0.6,0.5,0.3,0.6,0.5,1.0,0.7,0.8,0.9,0.0,0.6,0.3,0.9,0.3,0.9,0.5,0.7,0.5,0.1,0.1,0.3,0.7,0.8,0.1,0.0,0.7,0.5,1.0,0.3,0.8,0.7,0.7,0.2,0.9,0.5,0.6,0.1,0.5,0.5,0.0,0.2,0.7,0.9,0.1,0.9,0.3,0.2]]).to(
        device, dtype=torch.int64
    )
    trg =  torch.tensor([[0.5,0.6,0.0,0.9,0.9,0.4,0.4,0.9,0.1,0.7,0.8,0.7,1.0,0.5,0.6,0.5,0.9,0.7,0.2,0.4,0.6,0.7,0.4,0.2,0.3,0.3,0.9,1.0,0.0,0.5,0.5,0.6,0.1,0.6,0.1,1.0,0.8,0.4,0.2,0.6,0.9,0.2,0.1,0.5,0.0,0.5,0.3,0.9,0.5,0.0,0.9,0.4,0.4,0.5,0.7,0.9,0.1,0.9,0.0,0.2,0.6,0.8,0.7,0.1,0.6,0.2,0.2,0.8,0.7,0.2,0.1,0.2,0.6,0.8,0.6,0.4,0.8,0.8,0.9,0.7,0.8,0.4,0.5,0.1,0.7,0.9,0.2,0.3,0.0,0.7,0.0,0.1,0.7,0.8,0.9,0.7,0.6,0.3,0.7,0.7,0.2,0.1,0.3,0.7,0.3,0.8,0.2,0.1,0.8,0.9,0.2,0.4,0.5,0.5,0.9,0.9,0.3,0.7,0.1,0.6,0.7,0.2,0.6,0.9,0.8,0.7,0.0,0.4,0.1,0.6,0.5,0.1,0.8,0.7,0.9,0.7,0.5,0.7,0.8,0.8,0.2,0.5,0.3,0.4,0.8,0.4,0.1,0.3,0.4,0.3,0.4,0.7,0.4,0.7,0.9,0.2,0.8,0.3,0.8,0.3,0.8,0.7,0.3,0.4,0.4,0.6,0.1,0.3,0.6,0.5,0.9,0.7,0.3,0.6,0.5,0.3,0.4,0.2,0.8,0.3,0.1,0.9,0.9,0.6,0.1,0.4,0.2,0.4,0.8,0.9,0.1,0.4,0.8,0.5,0.4,0.8,0.9,1.0,0.1,0.8,0.8,0.8,0.8,0.8,0.3,0.1,1.0,0.2,0.9,0.2,0.9,0.7,0.9,1.0,0.4,0.2,0.5,0.4,0.3,0.2,0.1,0.1,0.8,0.7,0.0,0.3,1.0,1.0,0.0,0.5,0.0,0.5,0.6,0.8,0.2,0.4,0.0,0.8,0.5,0.8,0.6,0.3,0.4,0.7,0.9,0.0,0.8,0.7,0.9,0.9,0.2,0.3,0.3,0.9,0.3,0.3,0.3,0.6,0.8,0.5,0.5,0.0,0.5,0.8,1.0,0.4,1.0,0.3,0.5,0.5,0.6,0.6,0.7,0.1,0.3,0.6,0.4,0.2,0.8,1.0,0.6,0.9,0.7,0.5,0.1,0.7,0.6,1.0,0.4,0.9,0.3,0.6,0.1,1.0,0.8,0.7,0.7,0.5,0.0,0.6,0.5,1.0,0.6,0.9,0.8,0.9,0.7,1.0,0.9,1.0,0.3,0.2,0.5,0.3,0.8,0.1,0.9,0.6,0.9,0.9,0.3,0.4,0.1,0.6,0.0,0.0,0.2,0.2,0.9,0.9,0.6,1.0,0.2,0.7,1.0,0.8,1.0,0.2,0.3,0.3,0.9,0.5,0.1,0.2,0.5,0.9,0.1,0.5,0.2,1.0,0.7,0.4,0.2,0.1,0.4,0.4,0.7,0.8,0.3,0.6,0.0,1.0,0.8,1.0,0.1,0.2,0.9,0.4,0.8,0.0,0.0,1.0,0.1,0.3,0.0,0.7,0.6,0.9,0.4,0.4,0.9,0.4,0.8,0.7,0.7,0.5,0.3,0.6,0.5,0.5,0.5,0.9,0.8,0.4,0.8,0.6,0.4,0.2,0.9,1.0,0.8,0.2,0.2,0.8,0.9,0.7,0.1,0.8,0.7,0.3,0.1,0.2,0.3,0.6,0.6,0.6,0.7,0.4,0.1,0.9,0.5,0.5,0.5,0.4,0.6,0.2,0.7,0.6,0.3,0.3,0.2,0.4,0.2,0.9,0.9,0.9,0.7,0.8,0.3,0.0,0.4,0.1,0.9,0.6,0.3,0.0,0.7,0.1,0.8,0.6,0.3,0.6,0.8,0.2,0.1,0.4,0.8,0.9,1.0,0.7,0.8,0.1,0.4,0.1,0.4,0.9,0.4,0.6,0.7,0.2,0.5,0.6,0.8,0.6,0.6,0.9,0.7,0.4,0.3,0.5,0.1,0.8,0.9,0.4,0.0,0.4,0.0,0.3,0.6,0.8,0.1,0.4,0.1,0.6,0.7,0.1,0.0,0.0,0.0,0.8,0.7,0.6,0.8,0.6,0.9,0.1,0.4,0.0,0.4,0.0,0.4,0.7,0.5,0.1,0.9,0.3,0.3,0.1,0.3,0.6,0.6,0.8,0.8,0.9,0.2,0.0,0.6,0.3,1.0,0.6,0.7,1.0,1.0,0.9,0.4,0.1,0.6,0.9,0.1,0.1,0.1,0.2,0.5,0.0,0.8,0.5,0.0,0.8,0.4,0.1,0.2,0.2,0.8,0.9,0.6,0.3,0.2,0.5,0.0,0.1,0.1,0.8,0.9,1.0,0.8,0.2,0.8,0.3,0.8,0.2,0.0,0.1,1.0,0.7,0.1,0.8,0.2,0.5,0.3,0.6,0.1,0.7,0.7,0.5,0.2,0.3,0.5,0.5,1.0,0.2,0.3,0.4,0.1,0.1,0.7,1.0,0.7,0.6,0.9,1.0,0.4,0.8,0.1,0.4,0.1,0.9,0.7,0.4,0.0,0.0,0.3,0.3,0.5,0.6,0.3,0.8,0.5,0.3,0.1,0.9,0.5,0.1,0.3,0.9,0.4,0.3,0.4,0.2,0.9,0.5,0.4,0.9,0.8,0.9,0.9,0.9,0.6,0.6,0.3,0.4,0.3,0.3,0.4,0.4,0.2,0.3,0.7,0.1,0.4,0.1,0.7,0.2,0.7,0.7,0.1,0.3,1.0,0.4,0.4,0.0,0.1,0.4,0.6,0.9,0.5,0.1,0.6,0.9,0.1,0.2,0.4,0.5,0.5,0.1,0.7,0.0,0.1,1.0,0.6,0.1,0.5,0.7,0.2,0.7,0.1,0.1,0.5,0.5,0.2,0.7,0.0,0.9,0.3,0.2,0.9,0.2,0.2,0.5,0.5,0.6,0.3,0.4,0.9,0.4,0.5,0.8,0.1,0.4,0.5,0.9,0.5,0.4,0.3,1.0,0.7,0.5,0.1,0.0,0.3,0.0,0.5,0.5,0.9,0.6,0.3,0.7,0.1,0.9,0.1,0.9,0.1,0.8,0.0,0.9,0.0,0.0,0.7,0.6,1.0,0.5,0.9,0.7,0.4,0.5,0.6,0.3,0.6,0.9,0.4,0.3,0.3,1.0,0.2,1.0,0.3,0.7,0.9,0.8,0.8,0.7,0.6,0.6,0.8,0.5,0.3,0.4,0.5,0.1,0.3,0.4,0.0,0.2,0.8,0.3,1.0,0.5,0.0,0.7,0.9,0.3,0.3,0.9,0.9,0.5,0.0,0.0,0.6,0.7,0.6,0.5,0.1,0.8,0.3,0.3,0.1,0.7,0.0,0.6,0.0,0.1,0.9,0.1,0.4,0.1,0.5,1.0,0.3,0.2,0.8,0.6,0.3,0.5,0.3,0.1,0.9,0.1,0.9,0.9,0.1,0.8,0.7,0.8,0.3,0.5,1.0,0.1,0.7,0.4,0.7,0.7,0.9,0.9,1.0,0.3,0.8,0.3,0.3,0.5,0.2,0.6,0.4,0.5,0.7,0.8,0.9,0.8,0.9,0.2,0.0,0.5,0.2,1.0,0.7,0.4,0.1,0.6,0.6,0.0,0.4,0.6,0.6,0.4,0.1,0.7,1.0,0.1,0.4,0.3,0.9,0.1,0.0,0.1,0.6,0.1,1.0,0.1,0.3,0.3,0.4,0.3,0.8,0.2,0.5,0.1,0.3,0.8,0.7,0.0,0.4,0.5,0.2,0.0,0.5,0.8,0.2,0.6,0.9,0.8,0.9,0.5,0.7,0.5,0.9,0.9,0.3,0.5,0.3,1.0,0.8,0.7,0.9,0.6,0.6,0.5,0.8,0.2,0.7,0.6,0.3,0.1,0.9,0.2,0.4,0.9,0.3,0.2,0.5,0.5,0.9,0.2,1.0,0.9,0.8,0.2,0.2,1.0,0.4,0.4,0.6,0.8,0.3,0.2,0.6,0.0,0.5,0.9,0.6,0.3,0.4,0.8,0.5,0.6,0.7,0.6,0.0,0.1,0.3,0.7,0.4,0.1,0.2,0.7,0.2,0.3,0.8,0.2,0.4,0.2,1.0,1.0,0.7,0.8,0.2,0.5,0.3,0.5,0.4,0.6,0.5,0.3,0.6,0.5,1.0,0.7,0.8,0.9,0.0,0.6,0.3,0.9,0.3,0.9,0.5,0.7,0.5,0.1,0.1,0.3,0.7,0.8,0.1,0.0,0.7,0.5,1.0,0.3,0.8,0.7,0.7,0.2,0.9,0.5,0.6,0.1,0.5,0.5,0.0,0.2,0.7,0.9,0.1,0.9,0.3,0.2],
     [0.5,0.6,0.0,0.9,0.9,0.4,0.4,0.9,0.1,0.7,0.8,0.7,1.0,0.5,0.6,0.5,0.9,0.7,0.2,0.4,0.6,0.7,0.4,0.2,0.3,0.3,0.9,1.0,0.0,0.5,0.5,0.6,0.1,0.6,0.1,1.0,0.8,0.4,0.2,0.6,0.9,0.2,0.1,0.5,0.0,0.5,0.3,0.9,0.5,0.0,0.9,0.4,0.4,0.5,0.7,0.9,0.1,0.9,0.0,0.2,0.6,0.8,0.7,0.1,0.6,0.2,0.2,0.8,0.7,0.2,0.1,0.2,0.6,0.8,0.6,0.4,0.8,0.8,0.9,0.7,0.8,0.4,0.5,0.1,0.7,0.9,0.2,0.3,0.0,0.7,0.0,0.1,0.7,0.8,0.9,0.7,0.6,0.3,0.7,0.7,0.2,0.1,0.3,0.7,0.3,0.8,0.2,0.1,0.8,0.9,0.2,0.4,0.5,0.5,0.9,0.9,0.3,0.7,0.1,0.6,0.7,0.2,0.6,0.9,0.8,0.7,0.0,0.4,0.1,0.6,0.5,0.1,0.8,0.7,0.9,0.7,0.5,0.7,0.8,0.8,0.2,0.5,0.3,0.4,0.8,0.4,0.1,0.3,0.4,0.3,0.4,0.7,0.4,0.7,0.9,0.2,0.8,0.3,0.8,0.3,0.8,0.7,0.3,0.4,0.4,0.6,0.1,0.3,0.6,0.5,0.9,0.7,0.3,0.6,0.5,0.3,0.4,0.2,0.8,0.3,0.1,0.9,0.9,0.6,0.1,0.4,0.2,0.4,0.8,0.9,0.1,0.4,0.8,0.5,0.4,0.8,0.9,1.0,0.1,0.8,0.8,0.8,0.8,0.8,0.3,0.1,1.0,0.2,0.9,0.2,0.9,0.7,0.9,1.0,0.4,0.2,0.5,0.4,0.3,0.2,0.1,0.1,0.8,0.7,0.0,0.3,1.0,1.0,0.0,0.5,0.0,0.5,0.6,0.8,0.2,0.4,0.0,0.8,0.5,0.8,0.6,0.3,0.4,0.7,0.9,0.0,0.8,0.7,0.9,0.9,0.2,0.3,0.3,0.9,0.3,0.3,0.3,0.6,0.8,0.5,0.5,0.0,0.5,0.8,1.0,0.4,1.0,0.3,0.5,0.5,0.6,0.6,0.7,0.1,0.3,0.6,0.4,0.2,0.8,1.0,0.6,0.9,0.7,0.5,0.1,0.7,0.6,1.0,0.4,0.9,0.3,0.6,0.1,1.0,0.8,0.7,0.7,0.5,0.0,0.6,0.5,1.0,0.6,0.9,0.8,0.9,0.7,1.0,0.9,1.0,0.3,0.2,0.5,0.3,0.8,0.1,0.9,0.6,0.9,0.9,0.3,0.4,0.1,0.6,0.0,0.0,0.2,0.2,0.9,0.9,0.6,1.0,0.2,0.7,1.0,0.8,1.0,0.2,0.3,0.3,0.9,0.5,0.1,0.2,0.5,0.9,0.1,0.5,0.2,1.0,0.7,0.4,0.2,0.1,0.4,0.4,0.7,0.8,0.3,0.6,0.0,1.0,0.8,1.0,0.1,0.2,0.9,0.4,0.8,0.0,0.0,1.0,0.1,0.3,0.0,0.7,0.6,0.9,0.4,0.4,0.9,0.4,0.8,0.7,0.7,0.5,0.3,0.6,0.5,0.5,0.5,0.9,0.8,0.4,0.8,0.6,0.4,0.2,0.9,1.0,0.8,0.2,0.2,0.8,0.9,0.7,0.1,0.8,0.7,0.3,0.1,0.2,0.3,0.6,0.6,0.6,0.7,0.4,0.1,0.9,0.5,0.5,0.5,0.4,0.6,0.2,0.7,0.6,0.3,0.3,0.2,0.4,0.2,0.9,0.9,0.9,0.7,0.8,0.3,0.0,0.4,0.1,0.9,0.6,0.3,0.0,0.7,0.1,0.8,0.6,0.3,0.6,0.8,0.2,0.1,0.4,0.8,0.9,1.0,0.7,0.8,0.1,0.4,0.1,0.4,0.9,0.4,0.6,0.7,0.2,0.5,0.6,0.8,0.6,0.6,0.9,0.7,0.4,0.3,0.5,0.1,0.8,0.9,0.4,0.0,0.4,0.0,0.3,0.6,0.8,0.1,0.4,0.1,0.6,0.7,0.1,0.0,0.0,0.0,0.8,0.7,0.6,0.8,0.6,0.9,0.1,0.4,0.0,0.4,0.0,0.4,0.7,0.5,0.1,0.9,0.3,0.3,0.1,0.3,0.6,0.6,0.8,0.8,0.9,0.2,0.0,0.6,0.3,1.0,0.6,0.7,1.0,1.0,0.9,0.4,0.1,0.6,0.9,0.1,0.1,0.1,0.2,0.5,0.0,0.8,0.5,0.0,0.8,0.4,0.1,0.2,0.2,0.8,0.9,0.6,0.3,0.2,0.5,0.0,0.1,0.1,0.8,0.9,1.0,0.8,0.2,0.8,0.3,0.8,0.2,0.0,0.1,1.0,0.7,0.1,0.8,0.2,0.5,0.3,0.6,0.1,0.7,0.7,0.5,0.2,0.3,0.5,0.5,1.0,0.2,0.3,0.4,0.1,0.1,0.7,1.0,0.7,0.6,0.9,1.0,0.4,0.8,0.1,0.4,0.1,0.9,0.7,0.4,0.0,0.0,0.3,0.3,0.5,0.6,0.3,0.8,0.5,0.3,0.1,0.9,0.5,0.1,0.3,0.9,0.4,0.3,0.4,0.2,0.9,0.5,0.4,0.9,0.8,0.9,0.9,0.9,0.6,0.6,0.3,0.4,0.3,0.3,0.4,0.4,0.2,0.3,0.7,0.1,0.4,0.1,0.7,0.2,0.7,0.7,0.1,0.3,1.0,0.4,0.4,0.0,0.1,0.4,0.6,0.9,0.5,0.1,0.6,0.9,0.1,0.2,0.4,0.5,0.5,0.1,0.7,0.0,0.1,1.0,0.6,0.1,0.5,0.7,0.2,0.7,0.1,0.1,0.5,0.5,0.2,0.7,0.0,0.9,0.3,0.2,0.9,0.2,0.2,0.5,0.5,0.6,0.3,0.4,0.9,0.4,0.5,0.8,0.1,0.4,0.5,0.9,0.5,0.4,0.3,1.0,0.7,0.5,0.1,0.0,0.3,0.0,0.5,0.5,0.9,0.6,0.3,0.7,0.1,0.9,0.1,0.9,0.1,0.8,0.0,0.9,0.0,0.0,0.7,0.6,1.0,0.5,0.9,0.7,0.4,0.5,0.6,0.3,0.6,0.9,0.4,0.3,0.3,1.0,0.2,1.0,0.3,0.7,0.9,0.8,0.8,0.7,0.6,0.6,0.8,0.5,0.3,0.4,0.5,0.1,0.3,0.4,0.0,0.2,0.8,0.3,1.0,0.5,0.0,0.7,0.9,0.3,0.3,0.9,0.9,0.5,0.0,0.0,0.6,0.7,0.6,0.5,0.1,0.8,0.3,0.3,0.1,0.7,0.0,0.6,0.0,0.1,0.9,0.1,0.4,0.1,0.5,1.0,0.3,0.2,0.8,0.6,0.3,0.5,0.3,0.1,0.9,0.1,0.9,0.9,0.1,0.8,0.7,0.8,0.3,0.5,1.0,0.1,0.7,0.4,0.7,0.7,0.9,0.9,1.0,0.3,0.8,0.3,0.3,0.5,0.2,0.6,0.4,0.5,0.7,0.8,0.9,0.8,0.9,0.2,0.0,0.5,0.2,1.0,0.7,0.4,0.1,0.6,0.6,0.0,0.4,0.6,0.6,0.4,0.1,0.7,1.0,0.1,0.4,0.3,0.9,0.1,0.0,0.1,0.6,0.1,1.0,0.1,0.3,0.3,0.4,0.3,0.8,0.2,0.5,0.1,0.3,0.8,0.7,0.0,0.4,0.5,0.2,0.0,0.5,0.8,0.2,0.6,0.9,0.8,0.9,0.5,0.7,0.5,0.9,0.9,0.3,0.5,0.3,1.0,0.8,0.7,0.9,0.6,0.6,0.5,0.8,0.2,0.7,0.6,0.3,0.1,0.9,0.2,0.4,0.9,0.3,0.2,0.5,0.5,0.9,0.2,1.0,0.9,0.8,0.2,0.2,1.0,0.4,0.4,0.6,0.8,0.3,0.2,0.6,0.0,0.5,0.9,0.6,0.3,0.4,0.8,0.5,0.6,0.7,0.6,0.0,0.1,0.3,0.7,0.4,0.1,0.2,0.7,0.2,0.3,0.8,0.2,0.4,0.2,1.0,1.0,0.7,0.8,0.2,0.5,0.3,0.5,0.4,0.6,0.5,0.3,0.6,0.5,1.0,0.7,0.8,0.9,0.0,0.6,0.3,0.9,0.3,0.9,0.5,0.7,0.5,0.1,0.1,0.3,0.7,0.8,0.1,0.0,0.7,0.5,1.0,0.3,0.8,0.7,0.7,0.2,0.9,0.5,0.6,0.1,0.5,0.5,0.0,0.2,0.7,0.9,0.1,0.9,0.3,0.2]]).to(device, dtype=torch.int64)

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(
        device
    )
    out, attention = model(x, trg[:, :])
    print(out.shape)
    print(attention.shape)

'''