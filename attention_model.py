import torch 
import torch.nn as nn


















def forward(self,):
"""
dim=3 , key_length
"""

    N = query.shape[0]

    # queries_shape: (N, query_len, heads, heads_dim) # N is the input shape length..
    # keys shape: (N, key_len, self.heads, self.head_dim )
    # query shape: (N, key_len, self.heads, self.head_dim)
    # energy shape: (N, heads, query_len, key_len) 
    energy = torch.einsum("nqhd,nkhd")      # n,q,h,d



