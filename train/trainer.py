import torch
import torch.nn.functional as F

def train_step(data, sgc_model, gcn_model, att_module, optimizer, sim_high, new_edge_index, compute_normalized_aggregate, device):
    sgc_model.train(); gcn_model.train(); att_module.train()
    optimizer.zero_grad()
    logits_sgc = sgc_model(data.x, data.edge_index)
    probs_sgc = F.softmax(logits_sgc, dim=1)
    sim_low, _ = compute_normalized_aggregate(probs_sgc.detach().cpu().numpy(), new_edge_index.cpu(), data.num_nodes)
    sim_low = torch.tensor(sim_low).to(device).float()
    sim_high = sim_high.float().to(device)
    S, alpha_L, alpha_H = att_module(sim_low, sim_high)


    # row0, col0 = data.edge_index
    # edge_weight0 = S[row0, col0]
    # # find high-sim new edges
    # row1, col1 = torch.nonzero(S > 0.99995, as_tuple=True)
    # mask_no_self = row1 != col1
    # row1, col1 = row1[mask_no_self], col1[mask_no_self]
    # orig_edges_set = set(zip(row0.cpu().tolist(), col0.cpu().tolist()))
    # edges_np = torch.stack([row1, col1], dim=1).cpu().numpy()
    # import numpy as np
    # mask_new = ~np.array([tuple(e) in orig_edges_set for e in edges_np], dtype=bool)
    # row1, col1 = row1[mask_new], col1[mask_new]
    # edge_weight1 = S[row1, col1] if row1.numel()>0 else torch.tensor([], device=device)
    # edge_index = torch.cat([data.edge_index, torch.stack([row1, col1], dim=0)], dim=1) if row1.numel()>0 else data.edge_index
    # edge_weight = torch.cat([edge_weight0, edge_weight1], dim=0) if row1.numel()>0 else edge_weight0

    edge_index = data.edge_index
    n = data.num_nodes
    # edge_index是 [2, E], sim是全节点相似矩阵
    # 为边筛选对应权重
    row, col = edge_index
    edge_weight = S[row.cpu().numpy(), col.cpu().numpy()]  # numpy索引

    logits_gcn = gcn_model(data.x, edge_index, edge_weight=edge_weight)
    # Balanced softmax as in original
    train_labels = data.y[data.train_mask]
    num_classes = int(train_labels.max()) + 1
    cls_num = torch.bincount(train_labels, minlength=num_classes).float().to(device)

    logits_sgc_train = logits_sgc[data.train_mask]
    # 在 logits 上加上 log(cls_num)
    logits_sgc_bal = logits_sgc_train + torch.log(cls_num.unsqueeze(0) + 1e-12)
    loss_sgc = F.cross_entropy(logits_sgc_bal, train_labels)

    logits_gcn_train = logits_gcn[data.train_mask]
    logits_gcn_bal = logits_gcn_train + torch.log(cls_num.unsqueeze(0) + 1e-12)
    loss_gcn = F.cross_entropy(logits_gcn_bal, train_labels)


    # reg: reconstruct adjacency from logits (simple MSE)
    import numpy as np
    adj_reconstructed = torch.tensor(__import__('sklearn').metrics.pairwise.cosine_similarity(logits_gcn.detach().cpu().numpy()), device=device)
    row, col = data.edge_index
    A = torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=torch.ones(row.size(0), device=device), size=(data.num_nodes, data.num_nodes)).to_dense()
    loss_reg = F.mse_loss(adj_reconstructed, A)
    loss = loss_sgc + loss_gcn + 0.5 * loss_reg
    loss.backward(); optimizer.step()
    return loss.item(), gcn_model, edge_weight, edge_index, S
