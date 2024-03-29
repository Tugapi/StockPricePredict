from LSTMModel import LSTM
from dataset import getData
from options import args
import torch


def eval():
    model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, prediction_length=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    close_max, close_min, _, test_loader = getData(args.corpusFile, args.sequence_length, args.prediction_length, args.batch_size, args.train_ratio)
    for (x, label) in test_loader:
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(-1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())

    for i in range(len(preds)):
        print('预测值是%.2f,真实值是%.2f' % (
        preds[i][0] * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))

eval()