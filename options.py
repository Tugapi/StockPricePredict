import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--corpusFile', default='data/kedaxunfei.csv')


parser.add_argument('--gpu', default=0, type=int)  # gpu 卡号
parser.add_argument('--epochs', default=100, type=int)  # 训练轮数
parser.add_argument('--layers', default=2, type=int)  # LSTM层数
parser.add_argument('--input_size', default=8, type=int)  # 输入特征的维度
parser.add_argument('--hidden_size', default=32, type=int)  # 隐藏层的维度
parser.add_argument('--lr', default=0.0001, type=float)  # learning rate 学习率
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--sequence_length', default=40, type=int)  # sequence的长度，用前几天的数据来预测
parser.add_argument('--prediction_length', default=1, type=int)  # 模型预测天数
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--train_ratio', default=0.5, type=float)  # 用于训练的数据比例

parser.add_argument('--useGPU', default=False, type=bool)  # 是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool)  # 是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--save_file', default='model/stock.pth')  # 模型保存位置

# options for prediction
parser.add_argument('--used_days', default=100, type=int)  # 开始时已知的天数
parser.add_argument('--prediction_days', default=30, type=int)  # 要预测的天数



args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device