import random
from collections import defaultdict
from tqdm import tqdm

def anonymous_random_walk(graph, start_node, walk_length):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if not neighbors:
            break
        walk.append(random.choice(neighbors))
    node_to_id = {}
    anon_walk = []
    current_id = 1
    for node in walk:
        if node not in node_to_id:
            node_to_id[node] = current_id
            current_id += 1
        anon_walk.append(node_to_id[node])
    return tuple(anon_walk)

def collect_node_patterns(graph, walk_length=4, num_walks=100):
    node_patterns = defaultdict(list)
    pattern_set = set()
    for node in tqdm(graph.nodes()):
        for _ in range(num_walks):
            p = anonymous_random_walk(graph, node, walk_length)
            if len(p) == walk_length:
                node_patterns[node].append(p)
                pattern_set.add(p)
    return node_patterns, pattern_set
