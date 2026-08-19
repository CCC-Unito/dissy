from itertools import combinations
import krippendorff
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

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

def _sigma(data_matrix):
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

def sigma(data_source):
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
    return _sigma(data_matrix)


