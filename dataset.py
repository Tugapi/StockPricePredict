import pandas as pd
from pandas import read_csv
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms


def getData(filepath, sequence_length, prediction_length, batchsize, train_ratio):
    assert sequence_length > prediction_length, "输入长度应比预测长度长"
    stock_data = pd.read_csv(filepath, encoding='gbk', usecols=[18])
    drop_rows_index = stock_data[stock_data["收盘价"] < 0].index
    stock_data.drop(drop_rows_index, inplace=True)  # 去除无效行
    close_max = stock_data['收盘价'].max()
    close_min = stock_data['收盘价'].min()
    df = stock_data.apply(lambda x: (x - close_min) / (close_max - close_min))  # min-max标准化

    X=[]
    Y=[]
    for i in range(df.shape[0] - sequence_length - prediction_length):
        X.append(np.array(df.iloc[i:(i + sequence_length), ].values, dtype=np.float32))
        Y.append(np.array(df.iloc[(i + sequence_length):(i + sequence_length + prediction_length), ].values, dtype=np.float32)
)

    total_len = len(X)
    trainx, trainy = X[:int(train_ratio * total_len)], Y[:int(train_ratio * total_len)]
    testx, testy = X[int(train_ratio * total_len):], Y[:int(train_ratio * total_len)]
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchsize, shuffle=False, drop_last=True)
    return close_max, close_min, train_loader, test_loader

class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)
