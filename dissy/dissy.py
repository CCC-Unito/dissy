from itertools import combinations
import krippendorff
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def annotator_adjacency_list(data_matrix, verbose=False):
    """Build the annotator graph as an adjacency list.

    Returns a dict: vertex -> list of (neighbor, sign) tuples, where
    `sign` is +1 if the pairwise Krippendorff's alpha between vertex and
    neighbor is greater than the average alpha across all annotator pairs
    that share at least one annotation, and -1 otherwise. Annotator pairs
    with no shared (non-NaN) annotations, or with only one possible value
    across their shared annotations, get no edge.
    """
    n_annotators = data_matrix.shape[0]
    adjacency = {annotator: [] for annotator in range(n_annotators)}

    pairs = list(combinations(range(n_annotators), 2))

    # First pass: compute alpha for every valid pair, so we can get the
    # average before assigning signs (mirrors annotator_graph()'s two-pass
    # structure).
    pairwise_alpha = {}
    for annotator1, annotator2 in pairs:
        pair_matrix = np.array([data_matrix[annotator1], data_matrix[annotator2]])
        pair_matrix = pair_matrix[:, ~np.isnan(pair_matrix).any(axis=0)]

        if pair_matrix.shape[1] == 0:
            continue
        if len(set(pair_matrix[0]) | set(pair_matrix[1])) == 1:
            continue

        alpha = krippendorff.alpha(reliability_data=pair_matrix, level_of_measurement='nominal')
        pairwise_alpha[(annotator1, annotator2)] = alpha

    if not pairwise_alpha:
        return adjacency

    alpha_avg = np.mean(list(pairwise_alpha.values()))

    # Second pass: assign signs and populate the adjacency list.
    for (annotator1, annotator2), alpha in tqdm(pairwise_alpha.items(), disable=not verbose):
        sign = 1 if alpha > alpha_avg else -1
        adjacency[annotator1].append((annotator2, sign))
        adjacency[annotator2].append((annotator1, sign))

    return adjacency


def _sigma(data_matrix, verbose=False):
    adjacency_list = annotator_adjacency_list(data_matrix, verbose=verbose)

    # Only vertices that actually have at least one edge matter.
    active_vertices = [v for v, neighbors in adjacency_list.items() if neighbors]
    if not active_vertices:
        return 0

    # adj[v]: dict neighbor -> sign, for O(1) sign lookup and neighbor tests.
    adj = {v: dict(adjacency_list[v]) for v in active_vertices}

    # Order vertices by degree (ascending); this is what keeps the
    # intersection step below cheap on average.
    degree = {v: len(adj[v]) for v in active_vertices}
    order = {v: i for i, v in enumerate(sorted(active_vertices, key=lambda v: (degree[v], v)))}

    # Orient every edge from the lower-order (lower-degree) endpoint to the
    # higher-order endpoint.
    directed_adj = {v: [] for v in active_vertices}
    for u in active_vertices:
        for w in adj[u]:
            if order[u] < order[w]:
                directed_adj[u].append(w)

    total_cycles = 0
    balanced_cycles = 0

    # For each vertex u, every pair of its outgoing neighbors (v, w) that
    # are themselves connected forms a triangle (u, v, w), discovered
    # exactly once (since u has the lowest order in the triple).
    for u in tqdm(active_vertices, disable=not verbose):
        out_neighbors = directed_adj[u]
        for i in range(len(out_neighbors)):
            v = out_neighbors[i]
            for j in range(i + 1, len(out_neighbors)):
                w = out_neighbors[j]
                if w in adj[v]:
                    uv_sign = adj[u][v]
                    uw_sign = adj[u][w]
                    vw_sign = adj[v][w]

                    total_cycles += 1
                    if uv_sign * uw_sign * vw_sign == 1:
                        balanced_cycles += 1

    return balanced_cycles / total_cycles if total_cycles > 0 else 0


def annotator_graph(data_matrix, verbose=False):
    # create weighted undirected graph of annotators with krippenrodff's alpha as edge weight
    G = {}
    
    pos_e = 0
    neg_e = 0

    alpha_dist = []
    for annotator1, annotator2 in combinations(range(data_matrix.shape[0]), 2):
        pair_matrix = np.array([data_matrix[annotator1], data_matrix[annotator2]])
        pair_matrix = pair_matrix[:, ~np.isnan(pair_matrix).any(axis=0)]
        if pair_matrix.shape[1] == 0:
            continue
        if len(set(pair_matrix[0]) | set(pair_matrix[1])) == 1:
            continue
        alpha = krippendorff.alpha(reliability_data=pair_matrix, level_of_measurement='nominal',)
        alpha_dist.append(alpha)
    alpha_avg = np.mean(alpha_dist)

    for annotator1, annotator2 in tqdm(combinations(range(data_matrix.shape[0]), 2), total=len(list(combinations(range(data_matrix.shape[0]), 2))), disable=not verbose):
        pair_matrix = np.array([data_matrix[annotator1], data_matrix[annotator2]])
        # remove columns with NaN in either row
        pair_matrix = pair_matrix[:, ~np.isnan(pair_matrix).any(axis=0)]
        if pair_matrix.shape[1] == 0:
            continue
        # if there is only one possible value, continue
        if len(set(pair_matrix[0]) | set(pair_matrix[1])) == 1:
            continue
        alpha = krippendorff.alpha(reliability_data=pair_matrix, level_of_measurement='nominal',)

        if alpha > alpha_avg:
            #G.add_edge(annotator1, annotator2, weight=1)
            G[(annotator1, annotator2)] = 1
            G[(annotator2, annotator1)] = 1
            pos_e += 1
        else:
            G[(annotator1, annotator2)] = -1
            G[(annotator2, annotator1)] = -1
            neg_e += 1
    return G

def _sigma_legacy(data_matrix):
    G = annotator_graph(data_matrix)
    total_cycles = 0
    balanced_cycles = 0
    nodes = list(set([x for x,y in G.keys()]))
    for cycle in combinations(nodes, 3):
        a, b, c = cycle
        if not ((a,b) in G or (b,a) in G):
            continue
        if not ((b,c) in G or (c,b) in G):
            continue
        if not ((c,a) in G or (a,c) in G):
            continue
        ab_weight = G[(a,b)] if (a,b) in G else G[(b,a)]
        bc_weight = G[(b,c)] if (b,c) in G else G[(c,b)]
        ca_weight = G[(c,a)] if (c,a) in G else G[(a,c)]
        balanced = (ab_weight * bc_weight * ca_weight)==1
        total_cycles += 1
        if balanced:
            balanced_cycles += 1
    return balanced_cycles/total_cycles if total_cycles>0 else 0

def sigma(data_source, legacy=False):
    if isinstance(data_source, np.ndarray):
        data_matrix = data_source
    else:
        if type(data_source) == str:
            if os.path.isfile(data_source):
                try:
                    df = pd.read_csv(data_source)
                except Exception as e:
                    raise ValueError(f"Error reading file {data_source}: {e}")
            else:
                raise ValueError(f"File {data_source} does not exist.")            
        elif isinstance(data_source, pd.DataFrame):
            df = data_source
        try:
            data_matrix = df.pivot_table(
                index='instance',
                columns='annotator',
                values='label',
                fill_value=np.nan,
                aggfunc='sum'
            ).to_numpy().T
        except Exception as e:
            raise ValueError(f"Error processing data: {e}")
    if legacy:
        return _sigma_legacy(data_matrix)
    else:
        return _sigma(data_matrix)


