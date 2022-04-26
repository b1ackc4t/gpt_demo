import pandas as pd
import torch
from torch.utils import data
import pickle
from torch.utils.data import DataLoader
from util import tokenlize
from util import Word2Sequence


df1 = pd.read_csv("./data/sentiment_analysis_trainingset.csv")
df2 = pd.read_csv("./data/sentiment_analysis_validationset.csv")
df3 = pd.read_csv("./data/sentiment_analysis_testa.csv")
df1 = df1.loc[:, ['id', 'content', 'price_level']]
df2 = df2.loc[:, ['id', 'content', 'price_level']]
df3 = df3.loc[:, ['id', 'content', 'price_level']]
df_pre = df1.loc[:30000, ['id', 'content']]
df_train = df1.loc[30000:, ['id', 'content', 'price_level']]


class PreTrainDataSet(data.Dataset):
    def __init__(self, tokens_len):
        ws = pickle.load(open("./obj/ws.pcl", "rb"))
        self.data = [ws.transform(tokenlize(i), tokens_len) for i in df_pre['content']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class TrainDataSet(data.Dataset):
    def __init__(self, tokens_len):
        ws = pickle.load(open("./obj/ws.pcl", "rb"))
        self.data = [(ws.transform(tokenlize(text), tokens_len), label) for i, text, label in df_pre]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(batch):
    """
    控制每个batch返回什么
    :param batch: [batch_size, ]
    :return:
    """
    return torch.Tensor(batch)

if __name__ == '__main__':
    data = PreTrainDataSet(270)
    pickle.dump(data, open("./obj/pretrain_dataset.pcl", "wb"))
    loader = DataLoader(data, batch_size = 5, collate_fn=collate_fn)
    for i, item in enumerate(loader):
        print(i, item.size())
