import torch
import torch.nn as nn
import torch.nn.functional as F
from Mlayers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.nn import  SAGEConv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAEG_Net(torch.nn.Module):
    def __init__(self, args):
        super(SAEG_Net, self).__init__()
        self.args=args
        self.sage1 = SAGEConv(args.input_size, out_channels=args.hidden_size)
        self.sage2 = SAGEConv(args.hidden_size, out_channels=args.output_size)
        self.dropout = 0.5
        self.fcs = nn.ModuleList()
        for _ in range(args.input_size):
            self.fcs.append(
                nn.Sequential(
                    nn.Linear(args.seq_len * args.output_size, args.output_size),
                    nn.ReLU(),
                    nn.Linear(args.output_size, args.pre_len)
                )
            )

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        x = torch.flatten(x, start_dim=1)#（64，24*7）
        pred = []
        for idx in range(self.args.input_size):
            sub_pred = self.fcs[idx](x)
            pred.append(sub_pred)
        #pred = torch.stack(pred, dim=-1)
        #pred = pred.permute(0, 2, 1, 3)
        pred=torch.stack(pred)
        return pred


