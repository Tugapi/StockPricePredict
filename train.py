from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import LSTM
from options import args
from dataset import getData
import matplotlib.pyplot as plt

def train():
    # 保存训练配置
    argsDict = args.__dict__
    with open( 'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, prediction_length=args.prediction_length, dropout=args.dropout, batch_first=args.batch_first)
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    _, _, train_loader, _ = getData(args.corpusFile, args.sequence_length, args.prediction_length, args.batch_size, args.train_ratio)

    train_loss = []
    for i in range(args.epochs):
        total_loss = 0
        for (data, label) in train_loader:
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                label = label.squeeze(-1).cuda()
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1))
                label = label.squeeze(-1)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("%d epoch，total_loss: %f" % (i, total_loss))
        train_loss.append(total_loss)
    torch.save({'state_dict': model.state_dict()}, args.save_file)
    print('保存模型')
    # 绘制损失函数
    plt.figure()
    plt.plot(train_loss, 'b', label='train_loss')
    plt.ylabel('train_loss')
    plt.xlabel('epoch_num')
    plt.legend()
    plt.savefig("img/train_loss.jpg")

if __name__ == '__main__':
    train()