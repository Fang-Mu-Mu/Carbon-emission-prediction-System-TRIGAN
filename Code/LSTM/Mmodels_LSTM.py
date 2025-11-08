import torch
import torch.nn as nn
import torch.nn.functional as F
from Mlayers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import  SAGEConv
import torch.nn as nn
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ELM(nn.Module):
    def __init__(self, args):
        super(ELM, self).__init__()
        self.args = args
        self.final_layer=nn.ModuleList()
        for _ in range(args.input_size):
            self.final_layer.append(
                nn.Sequential(
                    nn.Linear(args.input_size * args.seq_len, args.hidden_size),
                    nn.ReLU(),
                    # 第三层：Dropout防止过拟合
                    nn.Dropout(0.5),
                    nn.Linear(args.hidden_size, args.output_size)
                )
            )

    def forward(self, x,args):
        pred=[]
        decoder_outputs=torch.flatten(x,start_dim=1)
        for idx in range(args.pre_len):
            sub_pred = self.final_layer[idx](decoder_outputs)  # b n pred_len
            pred.append(sub_pred)
        pred=torch.stack(pred)
        pred = pred.permute(2, 1, 0)
        return pred


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.input_size, args.hidden_size, args.num_layers, batch_first=True)
        self.final_layer = nn.ModuleList()
        for _ in range(args.input_size):
            self.final_layer.append(
                nn.Sequential(
                    nn.Linear(args.input_size, args.hidden_size),
                    nn.ReLU(),
                    # 第三层：Dropout防止过拟合
                    nn.Dropout(0.5),
                    nn.Linear(args.hidden_size, args.output_size)
                )
            )

    def forward(self, x,args):
        # 初始化隐藏状态h0, c0为全0向量
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        pred=[]
        x=torch.flatten(x,start_dim=1)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        for idx in range(args.pre_len):
            sub_pred = self.final_layer[idx](out)  # b n pred_len
            pred.append(sub_pred)
        pred=torch.stack(pred)
        pred = pred.permute(2, 1, 0)
        return pred
        # 将输入x和隐藏状态(h0, c0)传入LSTM网络

        # 取最后一个时间步的输出作为LSTM网络的输出

        return out




# 定义训练集和测试集的数据加载器
class MyDataset(Dataset):
    def __init__(self, data_X, data_Y):
        self.data_X = data_X
        self.data_Y = data_Y

    def __getitem__(self, index):
        x = self.data_X[index]
        y = self.data_Y[index]
        return x, y

    def __len__(self):
        return len(self.data_X)
