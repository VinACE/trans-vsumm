import torch
import torchvision  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys  
# VOCAB_SIZE = len(unique_tokens)
VOCAB_SIZE = 1024
NUM_EPOCHS = 10
HIDDEN_SIZE = 16
EMBEDDING_DIM = 256
BATCH_SIZE = 128
NUM_HEADS = 8
NUM_LAYERS = 6
NUM_LABELS = 1024
DROPOUT = .5
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda')
MAX_REVIEW_LEN = 1024


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src2, weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights


import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        output = src
        weights = []
        for mod in self.layers:
            output, weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)
        return output, weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerNet(nn.Module):
    def __init__(self, num_vocab, embedding_dim, hidden_size, nheads, n_layers, max_len, num_labels, dropout):
        super(TransformerNet, self).__init__()
        # embedding layer
        self.embedding = nn.Embedding(num_vocab, embedding_dim)
        # positional encoding layer
        self.pe = PositionalEncoding(embedding_dim, max_len = max_len)
        # encoder  layers
        enc_layer = TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
        self.encoder = TransformerEncoder(enc_layer, num_layers = n_layers)
        # final dense layer
        self.dense = nn.Linear(embedding_dim*max_len, num_labels)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        x = self.pe(x)
        x, w = self.encoder(x)
        x = x.reshape(x.shape[1], -1)
        x = self.dense(x)
        return x, w

"""

model = TransformerNet(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, MAX_REVIEW_LEN, NUM_LABELS, DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

%%time
loss_trace = []
for epoch in tqdm(range(NUM_EPOCHS)):
  current_loss = 0
  for i, (x, y) in enumerate(train_loader):
    x, y  = x.to(DEVICE), y.to(DEVICE)
    outputs, w = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_loss += loss.item()
  loss_trace.append(current_loss)


print(len(w))
print(w[0].shape)
"""

if __name__ == "__main__":
    # import pdb;pdb.set_trace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg =  torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    VOCAB_SIZE = 10
    NUM_LABELS = 10
    # model = TransformerNet(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(
    #     device
    # )
    model = TransformerNet(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, MAX_REVIEW_LEN, NUM_LABELS, DROPOUT).to(DEVICE)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # out = model(x, trg[:, :-1]):, :-1])[1]
    # print(out)
    # print("**************")
    # print(weight_mat_)
    print(model)

    outputs, w = model(x)
