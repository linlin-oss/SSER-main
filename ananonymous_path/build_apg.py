from collections import defaultdict, Counter
import numpy as np
import math

def is_suffix_prefix_match(p1, p2):
    return p1[2:] == p2[:-2]

def build_apg(pattern_set):
    patterns = list(pattern_set)
    index_map = {p: i for i, p in enumerate(patterns)}
    edges = defaultdict(list)
    # The original a.py attempted full pairwise; we keep structure placeholder
    return patterns, edges, index_map

def compute_node_distribution(node_patterns, index_map):
    node_dist = {}
    for node, patterns in node_patterns.items():
        counts = Counter()
        for p in patterns:
            counts[index_map[p]] += 1
        total = sum(counts.values())
        dist = np.zeros(len(index_map))
        for idx, c in counts.items():
            dist[idx] = c / total
        node_dist[node] = dist
    return node_dist

def propagate_on_apg(edges, dist_vec, alpha=0.85, max_iter=10):
    n = len(dist_vec)
    dist = dist_vec.copy()
    for _ in range(max_iter):
        new_dist = np.zeros(n)
        for i in range(n):
            neighbors = edges.get(i, [])
            if neighbors:
                weight = 1 / len(neighbors)
                for nb in neighbors:
                    new_dist[nb] += dist[i] * weight
            else:
                new_dist[i] += dist[i]
        dist = alpha * new_dist + (1 - alpha) * dist_vec
    return dist

def js_divergence(p, q):
    def kl_divergence(a,b):
        s=0.0
        for i in range(len(a)):
            if a[i]>0 and b[i]>0:
                s += a[i]*math.log(a[i]/b[i])
        return s
    m = 0.5*(p+q)
    return 0.5*kl_divergence(p,m) + 0.5*kl_divergence(q,m)
