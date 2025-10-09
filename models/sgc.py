# models/sgc.py
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree

class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        try:
            # 优先使用稀疏矩阵（GPU版本）
            from torch_sparse import SparseTensor
            adj_t = SparseTensor(row=col, col=row, value=norm, sparse_sizes=(num_nodes, num_nodes))
            for _ in range(self.K):
                x = adj_t.matmul(x)
        except RuntimeError as e:
            if "Not compiled with CUDA support" in str(e):
                # 使用稠密邻接矩阵（兼容 CPU / GPU）
                A = torch.zeros((num_nodes, num_nodes), device=x.device)
                A[row, col] = norm
                for _ in range(self.K):
                    x = torch.matmul(A, x)
            else:
                raise e

        return self.linear(x)
