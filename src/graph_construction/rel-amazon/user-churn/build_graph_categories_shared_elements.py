import numpy as np 
import pandas as pd 
import ast
import argparse
import os
import polars as pl
from collections import Counter
from collections import defaultdict
from html import unescape
from pprint import pprint

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import torch
import torch.nn.functional as F

from torch_geometric.utils import degree, to_undirected
from torch_geometric.data import Data 

def special_print(var, name):
    print()
    print('_'*150)
    print(f'\n {name} -> \n{var}\n')
    print('_'*150)
    print()

def return_density(n_nodes, n_edges):
    n_max_edges = (n_nodes*(n_nodes-1))/2
    density = (n_edges / n_max_edges) * 100
    return density

def build_similarity_map(train, data_name, min_support, min_lift):
    transactions = train[data_name].tolist()
    
    # 2. Initialize the TransactionEncoder
    te = TransactionEncoder()
    
    # 3. Fit and transform the data
    # te.fit(transactions) learns all unique categories
    # te.transform(transactions) converts the list of lists into a boolean array
    te_ary = te.fit(transactions).transform(transactions)
    
    # 4. Convert the array back into a pandas DataFrame
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Display the first few rows of the one-hot encoded data
    special_print(df_encoded.info(), 'df_encoded.info()')
    special_print(df_encoded.head(), 'df_encoded.head()')
    
    # Apply the Apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support = min_support, use_colnames=True)
    
    # Sort by support and display the top 10 frequent itemsets
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    special_print(frequent_itemsets, 'Frequent Itemsets')

    # Generate rules using a metric (e.g., 'lift') and a minimum threshold
    # Let's use lift > 1.2 and sort by lift to see the strongest rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold = min_lift)
    
    # Select and rename columns for a cleaner output
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    
    # Sort by lift to see the strongest associations first
    rules = rules.sort_values(by='lift', ascending=False)
    
    special_print(rules, f"Association Rules (Support > {min_support}, Lift > {min_lift}):")
    #rules.to_csv("rules.csv")

    strong_pairs = set()

    for _, row in rules.iterrows():
        # Convert frozensets to a list of strings and pair them up
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        # Store all A -> B pairs
        for ante in antecedents:
            for cons in consequents:
                strong_pairs.add((ante, cons))
                
    print(f"Total strong directional rules found: {len(strong_pairs)}")
    print()

    similarity_map = defaultdict(set)

    all_items = set() # Track all unique items encountered

    for item_a, item_b in strong_pairs:
        # A is similar to B
        similarity_map[item_a].add(item_b)

        #B is similar to A
        similarity_map[item_b].add(item_a)

        # Track all unique items
        all_items.add(item_a)
        all_items.add(item_b)

    for item in all_items:
        similarity_map[item].add(item)

    return similarity_map

def create_similarity_matrix(train, column_name, similarity_map):
    sets = [set(row) for row in train[column_name]]
    bags = [Counter(row) for row in train[column_name]]

    n = len(sets)
    similarity_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        if i % 100 == 0:
            print(f"Processing row {i}/{n}")
        for j in range(i, n):
            n_total_i = bags[i].total()
            n_total_j = bags[j].total()

            n_shared_elements_i = 0
            for item_i, count_i in bags[i].items():
                # An item is "shared" if its similarity group overlaps with the other user's set of items
                similar_to_item_i = similarity_map.get(item_i, {item_i})
                if not sets[j].isdisjoint(similar_to_item_i):
                    n_shared_elements_i += count_i
            
            n_shared_elements_j = 0
            for item_j, count_j in bags[j].items():
                similar_to_item_j = similarity_map.get(item_j, {item_j})
                if not sets[i].isdisjoint(similar_to_item_j):
                    n_shared_elements_j += count_j

            Ni = n_shared_elements_i / n_total_i if n_total_i > 0 else 0
            Nj = n_shared_elements_j / n_total_j if n_total_j > 0 else 0
            
            similarity_matrix[i, j] = max(Ni, Nj)
            similarity_matrix[j, i] = max(Ni, Nj)

    return similarity_matrix

def build_edge_index(similarity_matrix, threshold):
    edges = np.nonzero(np.triu(similarity_matrix > threshold, k=1))
    edge_index = np.stack(edges, axis=0)
    edge_index = torch.tensor(edge_index, dtype = torch.long)
    edge_index = to_undirected(edge_index)
    return edge_index

def find_threshold_for_target_density(similarity_matrix, n_nodes, target_density, tolerance=0.1, max_iter=100):
    """
    Finds the threshold that results in a graph density closest to the target_density.
    Uses binary search on the threshold value.
    """
    low = 0.0
    high = 1.0
    best_threshold = 0.0
    min_density_diff = float('inf')

    print(f"Searching for threshold to achieve target density: {target_density:.4f}")

    for i in range(max_iter):
        mid = (low + high) / 2
        
        # If search space is too small, stop.
        if high - low < 1e-6:
            print("Search space is too small, stopping.")
            break
            
        edge_index = build_edge_index(similarity_matrix, mid)
        n_edges = edge_index.shape[1] / 2
        density = return_density(n_nodes, n_edges)
        
        density_diff = abs(density - target_density)

        print(f"Iter {i+1}/{max_iter}: threshold={mid:.4f}, density={density:.4f}, diff={density_diff:.4f}")

        if density_diff < min_density_diff:
            min_density_diff = density_diff
            best_threshold = mid

        if min_density_diff <= tolerance:
            print(f"Found threshold with density difference within tolerance ({tolerance}).")
            break

        if density > target_density:
            low = mid
        else:
            high = mid
            
    final_edge_index = build_edge_index(similarity_matrix, best_threshold)
    final_n_edges = final_edge_index.shape[1] / 2
    final_density = return_density(n_nodes, final_n_edges)
    print(f"Search finished. Best threshold: {best_threshold:.4f} with density {final_density:.4f}")
    return best_threshold

def create_node_feature_table(edge_index, n_nodes):
    degrees = degree(edge_index[0], num_nodes=n_nodes, dtype=torch.long)
    x = F.one_hot(degrees).to(torch.float)
    return x

def return_data_partition_masks(nodes_id):
    train_len = int(len(nodes_id)*0.8)
    val_len = int(len(nodes_id)*0.1)

    train_mask = nodes_id < train_len
    val_mask = (nodes_id >= train_len) & (nodes_id < train_len + val_len)
    test_mask = nodes_id >= train_len + val_len

    masks = {
    'train_mask': train_mask,
    'val_mask': val_mask,
    'test_mask': test_mask
    }

    return masks

def save_data_object(data, directory_name, thrs, density, output_base_path):
    output_dir = os.path.join(output_base_path, directory_name)
    os.makedirs(output_dir, exist_ok = True)

    file_name = f'thr_{thrs:.2f}__{density:.2f}%.pt'

    output_path = os.path.join(output_dir, file_name)
    torch.save(data, output_path)
    print(f'Saved: {directory_name}/{file_name}')

def main(args):
    special_print(args.train, 'train')
    special_print(args.train[args.column_data].iloc[0], 'train first row')
    special_print(type(args.train[args.column_data].iloc[0]), 'type(train first row)')
    special_print(len(args.train[args.column_data].iloc[0]), 'len(train first row)')

    similarity_map = build_similarity_map(args.train, args.column_data, args.min_support, args.min_lift)

    print(f'\nSimilarity map: -> \n')
    pprint(similarity_map)
    special_print(len(similarity_map), 'len(similarity_map)')

    similarity_matrix = create_similarity_matrix(args.train, args.column_data, similarity_map)

    special_print(similarity_matrix, 'similarity_matrix')
    special_print(type(similarity_matrix), 'type(similarity_matrix)')
    special_print(similarity_matrix.shape, 'similarity_matrix.shape')

    n_nodes = len(args.train)
    
    if args.thr is not None:
        best_threshold = args.thr
        print(f"Using fixed threshold: {best_threshold}")
    else:
        best_threshold = find_threshold_for_target_density(
            similarity_matrix, 
            n_nodes, 
            args.target_density,
            tolerance=args.density_tolerance,
            max_iter=args.max_iter_search
        )

    edge_index = build_edge_index(similarity_matrix, best_threshold)

    n_edges = edge_index.shape[1] / 2

    density = return_density(n_nodes, n_edges)

    if args.target_density:
        special_print(args.target_density, 'target_density')
        
    special_print(best_threshold, 'best_threshold')
    special_print(edge_index, 'edge_index')
    special_print(edge_index.shape, 'edge_index.shape')
    special_print(density, 'final_density')

    #x = create_node_feature_table(edge_index, n_nodes)

    #y = torch.tensor(args.train['churn'].values, dtype=torch.long)
    #masks = return_data_partition_masks(args.train.index)
    #density = return_density(n_nodes, n_edges)

    #data = Data(x = x, edge_index = edge_index, y = y, masks = masks, density = round(density,2))

    #save_data_object(data, f'{args.column_data}', best_threshold, density, args.output_path)
    #special_print(data, 'data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process expanded training data into a graph using a feature.")
    parser.add_argument('output_path', type=str, help="Directory for output data")
    parser.add_argument('column_data', type=str, help="Name of the column data to use")     

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--target_density', type=float, help="Target density for the graph")
    group.add_argument('--thr', type=float, help="Threshold to generate the edges on the graph")

    parser.add_argument('--density_tolerance', type=float, default=5, help="Tolerance for target density")
    parser.add_argument('--max_iter_search', type=int, default=100, help="Max iterations for binary search of threshold")
    parser.add_argument('--base_data_path', type=str, default=r'data\2_intermediate\rel-amazon\tasks\user-churn', help="Directory base for all the expanded files")
    parser.add_argument('--kaggle', action='store_true', help="Flag to indicate whether the script will run on kaggle or not")
    parser.add_argument('--x_feature_actual_data', action='store_true', help="Flag to indicate that the node feature table will hold actual data")
    parser.add_argument('--use_subsection', action='store_true', help="Flag to indicate whether generate graph using only subsection")
    parser.add_argument('--sample_size', type=int, default=5, help="Number of samples to process from the train set")
    parser.add_argument('--min_support', type=float, default=0.01, help="")
    parser.add_argument('--min_lift', type=float, default=1.2, help="")

    args = parser.parse_args()

    train_file = os.path.join(args.base_data_path, args.column_data, 'expanded_train.csv')

    if args.use_subsection:
        #args.train = pd.read_csv(train_file, converters={args.column_data: parse_string_list}, index_col=0).head(args.sample_size)
        args.train = pd.read_csv(train_file, index_col=0).head(args.sample_size)
        args.train[args.column_data] = args.train[args.column_data].apply(unescape)
        args.train[args.column_data] = args.train[args.column_data].apply(ast.literal_eval)
    else:
        #args.train = pd.read_csv(train_file, converters={args.column_data: parse_string_list}, index_col=0)
        args.train = pd.read_csv(train_file, index_col=0)
        args.train[args.column_data] = args.train[args.column_data].apply(unescape)
        args.train[args.column_data] = args.train[args.column_data].apply(ast.literal_eval)

    os.makedirs(args.output_path, exist_ok=True)

    main(args)