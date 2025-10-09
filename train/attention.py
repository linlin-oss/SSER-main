import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityAttention(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.W_L = nn.Linear(num_nodes, hidden_dim, bias=True)
        self.W_H = nn.Linear(num_nodes, hidden_dim, bias=True)
        self.q = nn.Parameter(torch.randn(hidden_dim, 1) * 0.1)

    def forward(self, L: torch.Tensor, H: torch.Tensor):
        Wl = self.W_L(L)
        Wh = self.W_H(H)
        omega_L = torch.matmul(torch.tanh(Wl), self.q).squeeze(-1)
        omega_H = torch.matmul(torch.tanh(Wh), self.q).squeeze(-1)
        stacked = torch.stack([omega_L, omega_H], dim=1)
        alphas = F.softmax(stacked, dim=1)
        alpha_L = alphas[:,0].unsqueeze(1)
        alpha_H = alphas[:,1].unsqueeze(1)
        S = alpha_L * L + alpha_H * H
        S = 0.5 * (S + S.T)
        return S, alpha_L, alpha_H
