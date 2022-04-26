import math
import torch
import torch.nn as nn
from torch.nn import functional

class GPTConfig:
    """
    GPT模型的参数
    """
    def __init__(self, vocab_size, block_size, n_embed, tokens_len, pdrop, n_head, n_classify):
        self.vocab_size = vocab_size    # 词典大小
        self.block_size = block_size    # transformer decoder块数量
        self.n_embed = n_embed      # 词嵌入的维度
        self.tokens_len = tokens_len    # 序列长度
        self.pdrop = pdrop  # dropout率
        self.n_head = n_head    # multi头数量
        self.n_classify = n_classify    # 目标分类数量

class MaskedMultiSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.query = nn.Linear(config.n_embed, config.n_embed)
        self.key = nn.Linear(config.n_embed, config.n_embed)
        self.value = nn.Linear(config.n_embed, config.n_embed)
        self.register_buffer("masked", torch.tril(torch.ones(config.tokens_len, config.tokens_len)).view(1, 1, config.tokens_len, config.tokens_len))

    def forward(self, x: torch.Tensor):
        batch_size, tokens_len, n_embed = x.size()

        # 生成k q v
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # 划分出多头
        k = k.view(batch_size, tokens_len, self.n_head, n_embed // self.n_head).transpose(1, 2)
        q = q.view(batch_size, tokens_len, self.n_head, n_embed // self.n_head).transpose(1, 2)
        v = v.view(batch_size, tokens_len, self.n_head, n_embed // self.n_head).transpose(1, 2)

        # 对所有头求注意力权重
        att = (q @ k.transpose(-1, -2)) * math.sqrt(n_embed // self.n_head)
        # 添加蒙版
        att = att.masked_fill(self.masked == 0, float('-inf'))
        # 计算注意力权重
        att = functional.softmax(att, -1)
        y = (att @ v).transpose(1, 2).contiguous().view(batch_size, tokens_len, n_embed)

        return y


class PositionalEncoding(nn.Module):
    def __init__(self, config: GPTConfig):
        super(PositionalEncoding, self).__init__()
        ne = config.n_embed
        dropout = config.pdrop
        max_len = config.tokens_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, ne)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, ne, 2).float() * (-math.log(10000.0) / ne))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        return self.pe[:x.size(1), :].transpose(0, 1).repeat_interleave(5, dim=0)



class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
        self.att_model = MaskedMultiSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.lin1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.lin2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.drop1 = nn.Dropout(config.pdrop)

    def forward(self, x):
        x = x + self.att_model(x)
        x = self.ln1(x)
        x = x + self.drop1(self.lin2(functional.gelu(self.lin1(x))))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.positions_embed = PositionalEncoding(config)
        self.drop = nn.Dropout(config.pdrop)
        self.blocks = nn.ModuleList([Block(config) for i in range(config.block_size)])
        self.lin1 = nn.Linear(config.n_embed, config.n_embed)
        self.output_lin = nn.Linear(config.n_embed, config.n_classify)

    def forward(self, x):
        batch_size, token_size = x.size()
        token_e = self.tokens_embed(x)
        pos_e = self.positions_embed(x)
        x = token_e + pos_e
        for m in self.blocks:
            x = m(x)
        x = self.lin1(x)
        x = self.output_lin(x)
        x = functional.softmax(x, -1)
        return x






