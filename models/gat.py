import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

class WeightedGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, **kwargs):
        super().__init__(in_channels, out_channels, heads=heads, concat=concat, add_self_loops=False, **kwargs)
        self.edge_weight = None

    def forward(self, x, edge_index, edge_weight=None):
        self.edge_weight = edge_weight
        return super().forward(x, edge_index)

    def message(self, x_j, alpha, index, ptr, size_i):
        out = alpha.view(-1, self.heads, 1) * x_j
        if self.edge_weight is not None:
            w = self.edge_weight.view(-1, 1, 1)
            out = out * w
        return out

class WeightedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, concat=True):
        super().__init__()
        self.conv1 = WeightedGATConv(in_channels, hidden_channels, heads=heads, concat=concat)
        in_feats = hidden_channels * heads if concat else hidden_channels
        self.conv2 = WeightedGATConv(in_feats, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x
