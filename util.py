import re
import jieba
import pandas as pd
import pickle

def tokenlize(text):
    blacks = ['"', '\n']
    text = re.sub("|".join(blacks), "", text, flags=re.S)
    return jieba.lcut(text)

class Word2Sequence:
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    UNK = 0 # unknow
    PAD = 1 # padding
    def __init__(self):
        self.dict = {   # 词和数字的映射
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.idict = dict(zip(self.dict.values(), self.dict.keys()))
        self.count = {} # 词语频率

    def transform(self, words: list, max_len: int) -> list:
        """
        词序列转化为数字序列
        :param words:
        :param max_len:
        :return:
        """
        if len(words) <= max_len:
            words += [self.PAD_TAG] * (max_len - len(words))
        else:
            words = words[:max_len]
        return [self.dict.get(word, self.UNK) for word in words]

    def inverse_transform(self, nums: list, max_len: int) -> list:
        """
        数字序列转化为词序列
        :param nums:
        :param max_len:
        :return:
        """
        if len(nums) <= max_len:
            nums += [self.PAD_TAG] * (max_len - len(nums))
        else:
            nums = nums[:max_len]
        return [self.idict.get(num, self.UNK_TAG) for num in nums]

    def fit(self):
        df1 = pd.read_csv("./data/sentiment_analysis_trainingset.csv")
        df1 = df1['content']
        for text in df1:
            tmp = tokenlize(text)
            for word in tmp:
                self.count[word] = self.count.get(word, 0) + 1
        for num, word in zip(range(2, 2+len(self.count)),self.count):
            self.dict[word] = num
            self.idict[num] = word

if __name__ == '__main__':
    o = Word2Sequence()
    o.fit()
    pickle.dump(o, open("./obj/ws.pcl", "wb"))
