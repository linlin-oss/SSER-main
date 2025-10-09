import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from anonymous_path.random_walk import collect_node_patterns
from anonymous_path.build_apg import build_apg, compute_node_distribution, propagate_on_apg

class DynamicHighSimUpdater:
    def __init__(self, graph_nx, walk_length=5, num_walks=50, device='cpu', beta=1.0, alpha=0.5):
        self.G = graph_nx
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.device = device
        self.beta = beta
        self.alpha = alpha
        self.sim_high = None
        self.node_dist = None
        self.index_map = None
        self.pattern_set = None
        self.patterns = None
        self.edges = None
        self.A_orig = nx.to_numpy_array(self.G, dtype=float)

    def update_dynamic_graph(self, Z, threshold=0.7):
        Z_np = Z.detach().cpu().numpy()
        sim = cosine_similarity(Z_np)
        adj_orig_masked = self.A_orig * (sim >= threshold)
        sim[sim < threshold] = 0.0
        np.fill_diagonal(sim, 0.0)
        fused_adj = self.alpha * adj_orig_masked + (1 - self.alpha) * sim
        rows, cols = np.where(fused_adj > 0)
        edges = list(zip(rows.tolist(), cols.tolist()))
        G_dyn = nx.Graph()
        N = Z.size(0)
        G_dyn.add_nodes_from(range(N))
        G_dyn.add_edges_from(edges)
        connected_nodes = set(rows) | set(cols)
        all_nodes = set(range(N))
        isolated_nodes = all_nodes - connected_nodes
        for node in isolated_nodes:
            sim_scores = sim[node]
            sim_scores[node] = -1
            neighbor = int(np.argmax(sim_scores))
            edges.append((node, neighbor))
        G_dyn.add_edges_from(edges)
        self.G_dyn = G_dyn

    def compute_high_order_distribution(self):
        self.node_patterns, self.pattern_set = collect_node_patterns(self.G_dyn, self.walk_length, self.num_walks)
        self.patterns, self.edges, self.index_map = build_apg(self.pattern_set)
        self.node_dist = compute_node_distribution(self.node_patterns, self.index_map)
        # ensure all nodes present
        for node in self.G.nodes():
            if node not in self.node_dist:
                self.node_dist[node] = np.zeros(len(self.index_map))

    def compute_sim_high(self):
        nodes = list(self.node_dist.keys())
        dist_matrix = np.array([self.node_dist[node] for node in nodes])
        sim_high_dyn = cosine_similarity(dist_matrix)
        import torch
        return torch.tensor(sim_high_dyn, device=self.device)

    def ema_update(self, sim_high_dyn):
        if self.sim_high is None:
            self.sim_high = sim_high_dyn
        else:
            self.sim_high = (1 - self.beta) * self.sim_high + self.beta * sim_high_dyn
        return self.sim_high
