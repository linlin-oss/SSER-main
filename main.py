import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, from_scipy_sparse_matrix
from config import get_args
from models.sgc import SGC
from models.gcn import WeightedGCN
from models.mlp import MLP
from train.trainer import train_step
from train.dynamic_updater import DynamicHighSimUpdater
from train.attention import SimilarityAttention
from apg.random_walk import collect_node_patterns
from apg.build_apg import build_apg, compute_node_distribution
from utils.similarity import compute_normalized_aggregate
from utils.data_utils import stratified_split, make_imbalanced_least
from utils.metrics import evaluate_model, compute_class_acc
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_class_similarity_means_fast(S, labels):
    import numpy as np
    n = S.shape[0]
    same_sum = 0.0; diff_sum = 0.0
    same_cnt = 0; diff_cnt = 0
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if labels[i]==labels[j]:
                same_sum += S[i,j].item()
                same_cnt += 1
            else:
                diff_sum += S[i,j].item()
                diff_cnt += 1
    return (same_sum/same_cnt) if same_cnt>0 else 0.0, (diff_sum/diff_cnt) if diff_cnt>0 else 0.0

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f"{args.device}:0")
    dataset = Planetoid(root='data/', name=args.dataset)
    data = dataset[0].to(device)

    # ensure dense features
    try:
        data.x = data.x.to_dense()
    except:
        pass

    data = stratified_split(data, args.train_ratio, args.val_ratio, seed=args.seed)
    data_0 = data.clone()

    
    # 示例: 选择训练样本最少的前 k 个类作为少数类
    k = int((data.y.max() + 1) / 2)
    data = data.to(device)
    data, minorities = make_imbalanced_least(data, k=k, imbalance_ratio=args.imbalance_ratio, seed=args.seed)

    # construct initial APG high-sim from anonymous paths on original graph (may be slow)
    G = to_networkx(data, to_undirected=True)
    node_patterns, pattern_set = collect_node_patterns(G, walk_length=args.walk_length, num_walks=args.num_walks)
    patterns, edges, index_map = build_apg(pattern_set)
    node_dist = compute_node_distribution(node_patterns, index_map)
    # fill missing
    for node in G.nodes():
        if node not in node_dist:
            node_dist[node] = np.zeros(len(index_map))
    dist_matrix = np.array([node_dist[node] for node in sorted(node_dist.keys())])
    sim_high = cosine_similarity(dist_matrix)

    # models
    sgc_model = SGC(dataset.num_features, dataset.num_classes, K=args.K).to(device)
    gcn_model = WeightedGCN(dataset.num_features, args.hidden, dataset.num_classes).to(device)
    mlp_model = MLP(dataset.num_features, args.mlp_hidden, dataset.num_classes, dropout=args.dropout).to(device)
    att_module = SimilarityAttention(num_nodes=data.num_nodes, hidden_dim=args.att_hidden).to(device)
    learnable_sim_high_module = __import__('torch').nn.Parameter(torch.ones(data.num_nodes)).to(device) if False else None
    dynamic_updater = DynamicHighSimUpdater(G, walk_length=args.walk_length, num_walks=args.num_walks, device=device, beta=args.beta, alpha=args.alpha)

    optimizer = torch.optim.Adam(list(sgc_model.parameters()) + list(gcn_model.parameters()) + list(att_module.parameters()) + list(mlp_model.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)

    new_edge_index = data.edge_index
    same_means, diff_means = [], []
    for epoch in range(1, args.epochs+1):
        sim_high_tensor = torch.tensor(sim_high, device=device).float()
        loss, gcn_model, edge_weight, edge_index, S = train_step(data, sgc_model, gcn_model, att_module, optimizer, sim_high_tensor, new_edge_index, compute_normalized_aggregate, device)
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch}/{args.epochs} Loss={loss:.4f}")
            results = evaluate_model(gcn_model, data, edge_index, edge_weight)
            print(f" Train Acc: {results['train_mask']['accuracy']:.4f}, Val Acc: {results['val_mask']['accuracy']:.4f}, Test Acc: {results['test_mask']['accuracy']:.4f}")
        # periodic dynamic update (every 10 epochs)
        if epoch % 10 == 0:
            with torch.no_grad():
                z_emb = gcn_model(data.x, edge_index, edge_weight)
                dynamic_updater.update_dynamic_graph(z_emb, threshold=0.9)
                dynamic_updater.compute_high_order_distribution()
                sim_high_dyn = dynamic_updater.compute_sim_high()
                dynamic_updater.ema_update(sim_high_dyn)
                sim_high = dynamic_updater.sim_high.cpu().numpy()

    # final evaluation
    print('\n=== Final evaluation on test set ===')
    results = evaluate_model(gcn_model, data, edge_index, edge_weight)
    print(results)

if __name__ == '__main__':
    main()
