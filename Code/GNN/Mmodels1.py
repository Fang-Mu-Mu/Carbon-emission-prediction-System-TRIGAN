# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric_temporal.nn import STConv
from torch_geometric.nn import GATConv
from src.efficient_kan import KAN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STGCN(nn.Module):
    def __init__(self, args):
        super(STGCN, self).__init__()

        self.stgcn = STConv(
            num_nodes=1,  # 节点数
            in_channels=args.input_size,  # 每个节点的输入特征数
            hidden_channels=64,  # 隐藏层特征数
            out_channels=args.output_size,  # 每个节点的输出特征数
            kernel_size=3,  # 时间卷积核大小
            K=1,  # Chebyshev 滤波器大小
            normalization="sym",  # 图拉普拉斯矩阵的归一化方式
        )

    def forward(self, x, edge_index):
        # 确保 x 的形状是 (batch_size, seq_len, num_nodes, in_channels)
        x, edge_index = x.to(device), edge_index.to(device)
        x = self.stgcn(x, edge_index)
        return x

# class STGCN_MLP(nn.Module):
#     def __init__(self, args):
#         super(STGCN_MLP, self).__init__()
#         self.args = args
#         self.out_feats = 128
#         self.stgcn = STGCN(num_nodes=args.input_size, size=3, K=1)
#         self.fcs = nn.ModuleList()
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 32))  # 调整到固定大小
#         #self.e_kan = KAN([10, 10, 1])
#         for k in range(args.input_size):
#             self.fcs.append(nn.Sequential(
#                 nn.Linear(36 * 14, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, args.output_size)
#             ))
#
#
#     def forward(self, x, edge_index):
#         # x(batch_size, seq_len, input_size)
#         print(f"Input x shape: {x.shape}")
#         x = x.unsqueeze(3)
#         print(f"After unsqueeze: {x.shape}")
#         x = self.stgcn(x, edge_index)
#         print(f"After STGCN: {x.shape}")
#         x = x.unsqueeze(3)
#         x = self.stgcn(x, edge_index)
#         #print(2,x.shape)
#         preds = []
#         for k in range(x.shape[2]):
#             preds.append(self.fcs[k](torch.flatten(x[:, :, k, :], start_dim=1)))
#
#         pred = torch.stack(preds, dim=0)
#
#         return pred
