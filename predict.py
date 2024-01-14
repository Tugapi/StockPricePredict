from LSTMModel import LSTM
from dataset import getData
from options import args
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict(corpusFile=args.corpusFile, sequence_length=args.sequence_length, prediction_length=args.prediction_length, used_days=args.used_days, prediction_days=args.prediction_days):
    assert prediction_days % prediction_length == 0, "待预测天数要能整除模型输出长度"
    total_iter = int(prediction_days / prediction_length)

    model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, prediction_length=args.prediction_length)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    stock_data = pd.read_csv(corpusFile, encoding='gbk', usecols=[18])
    drop_rows_index = stock_data[stock_data["收盘价"] < 0].index
    stock_data.drop(drop_rows_index, inplace=True)  # 去除无效行
    X_ori = np.array(stock_data.iloc[0:used_days + prediction_days, ].values, dtype=np.float32)


    close_max = stock_data['收盘价'].max()
    close_min = stock_data['收盘价'].min()
    df = stock_data.apply(lambda x: (x - close_min) / (close_max - close_min))  # min-max标准化

    pred = []
    datas = [x for x in range(used_days, used_days + prediction_days)]
    for i in range(total_iter):
        X = np.array(df.iloc[used_days - sequence_length + i * prediction_length : used_days + i * prediction_length, ].values, dtype=np.float32)
        transform = transforms.ToTensor()
        input = transform(X)
        Y_pred = model(input).squeeze(0)
        pred.extend(Y_pred.detach().numpy() * (close_max - close_min) + close_min)

    # 绘制结果
    plt.figure()
    plt.plot(X_ori, 'b', label='ground_truth')
    plt.plot(datas, pred, 'r', label='prediction_result')
    plt.ylabel('price')
    plt.xlabel('days')
    plt.legend()
    plt.savefig("img/prediction_result.jpg")

if __name__ == '__main__':
    predict()