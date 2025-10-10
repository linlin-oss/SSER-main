import torch
import random
from torch_geometric.utils import to_networkx

def stratified_split(data, train_ratio, val_ratio, seed=42):
    torch.manual_seed(seed)
    num_nodes = data.x.size(0)
    num_classes = int(data.y.max().item()) + 1
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.size(0))]
        n_total = idx.size(0)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train+n_val]
        test_idx = idx[n_train+n_val:]
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def make_imbalanced_least(data, k=2, imbalance_ratio=0.5, seed=42):
    torch.manual_seed(seed)
    train_mask = data.train_mask.clone()
    test_mask = data.test_mask.clone()
    train_idx = torch.where(train_mask)[0]
    num_classes = int(data.y.max().item()) + 1
    class_counts = []
    for c in range(num_classes):
        count = (data.y[train_idx] == c).sum().item()
        class_counts.append((c, count))
    class_counts.sort(key=lambda x: x[1])
    minority_classes = [c for c,_ in class_counts[:k]]
    for c in minority_classes:
        class_idx = train_idx[data.y[train_idx] == c]
        per_class_train = len(class_idx)
        num_to_remove = int(per_class_train * imbalance_ratio)
        if num_to_remove <= 0:
            continue
        import random
        remove_nodes = random.sample(class_idx.tolist(), num_to_remove)
        train_mask[remove_nodes] = False
        test_mask[remove_nodes] = True
        
        print(f"类别 {c}: 原本 {per_class_train}, 移除 {num_to_remove}, 剩余 {per_class_train - num_to_remove}")

    data.train_mask = train_mask
    data.test_mask = test_mask
    return data, minority_classes

def get_data(dataset_name):
    if dataset_name in ['cora', 'citeseer']:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='data/', name=dataset_name)
    elif dataset_name == 'photo':
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root='data/', name=dataset_name)
    elif dataset_name == 'blogcatalog' or dataset_name == 'flickr':
        from torch_geometric.datasets import AttributedGraphDataset
        dataset = AttributedGraphDataset(root='data/', name=dataset_name)
    elif dataset_name == 'wisconsin':
        from torch_geometric.datasets import WebKB
        dataset = WebKB(root='data/', name=dataset_name)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def pre_dealheterophily(data, device):
    """
    异质性图结构预处理：
    - 对低度节点执行邻边降权；
    - 对中高节点执行低相似度边降权；
    - 基于特征相似度添加 KNN 边。
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import scipy.sparse as sp
    from collections import defaultdict
    from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

    # === 1. 参数设置 ===
    knn_k = 30
    knn_k_mid_high = 10
    low_deg_edge_weight = 0.0
    mid_high_edge_weight = 0.0
    new_knn_edge_weight = 1.0
    low_deg_keep_ratio = 0.6
    low_sim_ratio = 0.4
    low_ratio = 0.2  

    # === 2. 相似度计算（GPU） ===
    X = data.x.to_dense().to(device)
    X_norm = F.normalize(X, p=2, dim=1)
    cos_sim = torch.matmul(X_norm, X_norm.T)  # 余弦相似度矩阵（GPU）

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tolil()

    degrees = np.array(A.sum(axis=1)).flatten()
    sorted_idx = np.argsort(degrees)
    low_deg_nodes = sorted_idx[:int(low_ratio * num_nodes)]
    mid_high_nodes = sorted_idx[int(low_ratio * num_nodes):]


    rows, cols = A.nonzero()
    rows = np.array(rows)
    cols = np.array(cols)
    low_deg_mask = np.isin(rows, low_deg_nodes)
    low_rows, low_cols = rows[low_deg_mask], cols[low_deg_mask]
    non_diag_mask = low_rows != low_cols
    low_rows, low_cols = low_rows[non_diag_mask], low_cols[non_diag_mask]

    low_sims = cos_sim[low_rows, low_cols].cpu().numpy()
    group = defaultdict(list)
    for i, j, sim in zip(low_rows, low_cols, low_sims):
        group[i].append((j, sim))

    for i, pairs in group.items():
        pairs.sort(key=lambda x: -x[1])  # 相似度从高到低
        topk = int(len(pairs) * low_deg_keep_ratio)
        for j, _ in pairs[topk:]:
            A[i, j] = low_deg_edge_weight
            A[j, i] = low_deg_edge_weight

    mid_high_mask = np.ones(num_nodes, dtype=bool)
    mid_high_mask[low_deg_nodes] = False
    low_deg_mask_bool = ~mid_high_mask

    adj_dict = defaultdict(list)
    row, col = A.nonzero()
    for i, j in zip(row, col):
        if mid_high_mask[i]:
            adj_dict[i].append(j)

    rows_to_update, cols_to_update = [], []
    for i in mid_high_nodes:
        nbrs = adj_dict[i]
        if len(nbrs) == 0:
            continue
        sims = cos_sim[i, nbrs].cpu().numpy()
        k = max(1, int(low_sim_ratio * len(sims)))
        low_sim_idx = np.argsort(sims)[:k]
        for j in np.array(nbrs)[low_sim_idx]:
            if low_deg_mask_bool[j] and A[i, j] != 1.0:
                continue
            rows_to_update.append(i)
            cols_to_update.append(j)

    A[rows_to_update, cols_to_update] = mid_high_edge_weight
    A[cols_to_update, rows_to_update] = mid_high_edge_weight

    # === 6. 基于相似度的 KNN 加边 ===
    same_class_count, diff_class_count = 0, 0
    same_class_cnt = np.zeros(num_nodes, dtype=int)
    diff_class_cnt = np.zeros(num_nodes, dtype=int)

    for node_set, k in [(low_deg_nodes, knn_k), (mid_high_nodes, knn_k_mid_high)]:
        for i in node_set:
            sim_row = cos_sim[i].cpu().numpy()
            topk_idx = np.argsort(-sim_row)[1:k + 1]
            for j in topk_idx:
                A[i, j] = A[j, i] = new_knn_edge_weight
                if data.y[i] == data.y[j]:
                    same_class_count += 1
                    same_class_cnt[i] += 1
                else:
                    diff_class_count += 1
                    diff_class_cnt[i] += 1

    print(f"KNN 加边 | 同类边: {same_class_count}，异类边: {diff_class_count}")

    # === 7. 转为 PyG 格式并返回 ===
    new_edge_index, new_edge_weight = from_scipy_sparse_matrix(A)
    data.edge_index = new_edge_index.to(device)
    data.edge_weight = new_edge_weight.to(dtype=torch.float32).to(device)
    data.y = data.y.to(device)

    return data
