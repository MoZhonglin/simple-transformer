import math
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# configs
device = "cuda"
epochs = 100

# dataset
sentences = [
    # enc_input, dec_input, dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# 源语言和目标语言的vocab要分开建立
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_idx2word = {i: w for i, w in enumerate(src_vocab)}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
tgt_idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

# 源句子与目标句子
src_len = 5
tgt_len = 6

# Transformer Parameters
d_model = 512   # embedding size, token & position
d_ff = 2048     # FFN dimension，线性层用来做特征提取
d_k = d_v = 64  # K, Q维度需要相等，为了方便，可以让V的维度也相等
n_layers = 6    # number of encoder & decoder blocks
n_heads = 8


# 数据构建
def make_data(sentences):
    """ make data """
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
print(enc_inputs)
print(dec_inputs)
print(dec_outputs)


class MyDataSet(data.Dataset):
    """ dataset """
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        """ init """
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        """ len """
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        """ get item magic method """
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    """ get attn subsequence """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    """ scaled dot product attention """
    def __init__(self):
        """ init """
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """ forward """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    """ multi head attention """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PositionWiseFeedForwardNet(nn.Module):
    """ position wise feed forward net """
    def __init__(self):
        """ init """
        super(PositionWiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """ forward """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)


class EncoderLayer(nn.Module):
    """ encoder layer """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositionWiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """ forward """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    """ decoder layer """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PositionWiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """ forward """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


# Transformer
class PositionalEncoding(nn.Module):
    """ positional encoding """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """ init """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ forward """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):
    """ encoder """
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """ forward """
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    """ decoder """
    def __init__(self):
        """ init """
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """ forward """
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)
        dec_self_attn_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)
        dec_self_attn_mask = torch.gt((dec_self_attn_mask + dec_self_attn_subsequence_mask), 0).to(device)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    """ transformer """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

model = Transformer().to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in range(epochs):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        enc_inputs, dec_inputs, dec_outputs = (enc_inputs.to(device), dec_inputs.to(device),
                                               dec_outputs.to(device))
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = loss_fn(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat([dec_input.to(device),
                               torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["E"]:
            terminal = True
    greedy_dec_predict = dec_input[:, 1:]
    return greedy_dec_predict


loader = data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, False)
enc_inputs, _, _ = next(iter(loader))
print(enc_inputs)
for i in range(len(enc_inputs)):
    greedy_dec_predict = greedy_decoder(model, enc_inputs[i].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
    print(enc_inputs[i], '->', greedy_dec_predict.squeeze())
    print([src_idx2word[t.item()] for t in enc_inputs[i]], '->',
          [tgt_idx2word[n.item()] for n in greedy_dec_predict.squeeze()])
