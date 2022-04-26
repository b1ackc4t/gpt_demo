import torch
from torch.utils.data import DataLoader
from dataset import PreTrainDataSet, collate_fn
from model import *
from util import Word2Sequence
import pickle
from torch.nn import functional

ws = pickle.load(open("./obj/ws.pcl", "rb"))
pretrain_dataset = pickle.load(open("./obj/pretrain_dataset.pcl", "rb"))
batch_size = 5
loader = DataLoader(pretrain_dataset, batch_size = batch_size, collate_fn=collate_fn)
config = GPTConfig(len(ws.dict), 3, 200, 270, 0.1, 8, len(ws.dict))
gpt = GPT(config)
for i, x in enumerate(loader):
    print(i, x.size())
    with torch.set_grad_enabled(True):
        x.type(torch.long)
        y = gpt(x.long())
        # loss = functional.cross_entropy(x.view(-1, x.size(-1)), x.view(-1))

        print()
    break