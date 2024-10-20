import sys

import networkx

import pandas as pd
import itertools
import numpy as np
from scipy.special import gammaln
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import math
from networkx.drawing.nx_pydot import write_dot
import os
import graphviz

# def write_gph(dag, idx2names, filename):
#     with open(filename, 'w') as f:
#         for edge in dag.edges():
#             f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

    #We are using a Uniform Dirichlet Prior

    data = read_csv(infile)

    # Initialize the graph with no edges (nodes only)
    graph = nx.DiGraph()
    graph.add_nodes_from(data.columns)  # Add the variable names as nodes

    # Track the start time
    start_time = time.time()

    # Choose a structure learning algorithm below.
    best_graph = local_search(graph, data)
    # best_graph = hill_climbing_search(graph, data)
    # best_graph = simulated_annealing_search(graph, data)

    # Track the end time
    end_time = time.time()

    # Calculate the time taken and print it
    elapsed_time = end_time - start_time
    print(f"Structure learning completed in {elapsed_time:.2f} seconds.")

    save_gph(best_graph, outfile)
    pass

# Function to read the .csv file and return data as a pandas DataFrame
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

# def local_search(graph, data, max_parents=100, max_iter=100000, patience=100000, restarts=10):
#     """
#     Perform improved local search with random restarts and parent limit to learn the structure of the Bayesian network.
#     Args:
#         graph: The initial directed graph (empty graph).
#         data: The dataset (pandas DataFrame).
#         max_parents: Maximum number of parents each node can have.
#         max_iter: Maximum number of iterations.
#         patience: Stop if no improvement after this many iterations.
#         restarts: Number of random restarts to escape local optima.
#     Returns:
#         A directed graph representing the learned structure.
#     """
#     best_graph = graph.copy()
#     best_score = bayesian_score(best_graph, data)

#     for restart in range(restarts):
#         print(f"Random restart {restart + 1}/{restarts}")
#         current_graph = random_initialize_graph(graph, data)
#         current_score = bayesian_score(current_graph, data)
#         iteration = 0
#         no_improvement_iters = 0
#         improving = True

#         while improving and iteration < max_iter and no_improvement_iters < patience:
#             improving = False

#             # Try adding, removing, or reversing edges between pairs of nodes
#             for u, v in itertools.permutations(graph.nodes(), 2):
#                 # Skip if the maximum number of parents is reached
#                 if len(list(current_graph.predecessors(v))) >= max_parents:
#                     continue
                
#                 # Randomly decide to add, remove, or reverse edges
#                 candidate_moves = []
#                 if current_graph.has_edge(u, v):
#                     candidate_moves.append(('remove', u, v))
#                 else:
#                     candidate_moves.append(('add', u, v))

#                 # Shuffle candidate moves for randomized greedy selection
#                 random.shuffle(candidate_moves)

#                 for move in candidate_moves:
#                     if move[0] == 'add':
#                         current_graph.add_edge(u, v)
#                     elif move[0] == 'remove':
#                         current_graph.remove_edge(u, v)

#                     if nx.is_directed_acyclic_graph(current_graph):
#                         new_score = bayesian_score(current_graph, data)
#                         if new_score > current_score:
#                             current_score = new_score
#                             improving = True
#                             no_improvement_iters = 0
#                         else:
#                             # Revert changes if not improving
#                             if move[0] == 'add':
#                                 current_graph.remove_edge(u, v)
#                             elif move[0] == 'remove':
#                                 current_graph.add_edge(u, v)

#             # Update iteration counters
#             iteration += 1
#             if not improving:
#                 no_improvement_iters += 1

#         # Update the best score and graph after each restart
#         if current_score > best_score:
#             best_score = current_score
#             best_graph = current_graph.copy()

#     return best_graph

# def random_initialize_graph(graph, data):
#     """
#     Initialize a graph with random edges while ensuring it's a DAG.
#     Args:
#         graph: The graph to initialize.
#         data: The dataset (pandas DataFrame).
#     Returns:
#         A directed acyclic graph with random edges.
#     """
#     random_graph = graph.copy()
#     node_list = list(graph.nodes())
#     random.shuffle(node_list)

#     for u, v in itertools.permutations(node_list, 2):
#         if random.random() < 0.1:  # 10% chance to add an edge randomly
#             random_graph.add_edge(u, v)
#         if not nx.is_directed_acyclic_graph(random_graph):
#             random_graph.remove_edge(u, v)

#     return random_graph

def local_search(graph, data):
    """
    Perform hill-climbing search to learn the structure of the Bayesian network.
    Args:
        graph: The initial directed graph (empty graph).
        data: The dataset (pandas DataFrame).
    Returns:
        A directed graph representing the learned structure.
    """
    best_graph = graph.copy()
    best_score = bayesian_score(best_graph, data)
    improving = True

    # Perform hill-climbing until no further improvement is possible
    while improving:
        improving = False
        current_score = best_score

        print("Local search in progress. Current bayesian score: ", current_score)

        # Try adding, removing, or reversing edges between each pair of nodes
        for u, v in itertools.permutations(graph.nodes(), 2):
            if best_graph.has_edge(u, v):
                # Try removing the edge
                best_graph.remove_edge(u, v)
                new_score = bayesian_score(best_graph, data)
                if new_score > current_score:
                    current_score = new_score
                    improving = True
                else:
                    # Revert change if it doesn't improve the score
                    best_graph.add_edge(u, v)
            else:
                # Try adding or reversing the edge
                best_graph.add_edge(u, v)
                new_score = bayesian_score(best_graph, data)
                if new_score > current_score and nx.is_directed_acyclic_graph(best_graph):
                    current_score = new_score
                    improving = True
                else:
                    # Revert change if it doesn't improve the score
                    best_graph.remove_edge(u, v)

        # Update the best score and graph if any improvement was made
        if improving:
            best_score = current_score

    return best_graph

def simulated_annealing_search(graph, data, initial_temp=100, cooling_rate=0.99, max_iter=10000):
    """
    Perform simulated annealing search to learn the structure of the Bayesian network.
    Args:
        graph: The initial directed graph (empty graph).
        data: The dataset (pandas DataFrame).
        initial_temp: Initial temperature for simulated annealing.
        cooling_rate: Rate at which the temperature decreases.
        max_iter: Maximum number of iterations.
    Returns:
        A directed graph representing the learned structure.
    """
    current_graph = graph.copy()
    current_score = bayesian_score(current_graph, data)
    best_graph = current_graph.copy()
    best_score = current_score

    temp = initial_temp
    iteration = 0

    while iteration < max_iter and temp > 1e-6:  # Run until max iterations or temperature is too low
        iteration += 1

        # Randomly select a pair of nodes to modify the graph
        u, v = random.sample(list(graph.nodes()), 2)
        
        # Randomly decide whether to add, remove, or reverse an edge
        if current_graph.has_edge(u, v):
            current_graph.remove_edge(u, v)  # Remove the edge
        else:
            current_graph.add_edge(u, v)  # Add or reverse the edge
        
        # Ensure the graph remains a DAG
        if nx.is_directed_acyclic_graph(current_graph):
            new_score = bayesian_score(current_graph, data)

            # Accept the new graph if it's better or with a probability based on temperature
            if new_score > current_score or random.random() < math.exp((new_score - current_score) / temp):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_graph = current_graph.copy()
            else:
                # Revert the change if not accepted
                if current_graph.has_edge(u, v):
                    current_graph.remove_edge(u, v)
                else:
                    current_graph.add_edge(u, v)

        # Cool down the temperature
        temp *= cooling_rate

    return best_graph

# def simulated_annealing_search(graph, data, initial_temp=100, cooling_rate=0.99, max_iter=10000):
#     """
#     Perform simulated annealing search to learn the structure of the Bayesian network.
#     Args:
#         graph: The initial directed graph (empty graph).
#         data: The dataset (pandas DataFrame).
#         initial_temp: Initial temperature for simulated annealing.
#         cooling_rate: Rate at which the temperature decreases.
#         max_iter: Maximum number of iterations.
#     Returns:
#         A directed graph representing the learned structure.
#     """
#     # Initialize the graph with all variables as nodes
#     current_graph = graph.copy()
#     current_score = bayesian_score(current_graph, data)
#     best_graph = current_graph.copy()
#     best_score = current_score

#     temp = initial_temp
#     iteration = 0

#     while iteration < max_iter and temp > 1e-6:  # Run until max iterations or temperature is too low
#         iteration += 1

#         # Randomly select a pair of nodes to modify the graph
#         u, v = random.sample(list(graph.nodes()), 2)
        
#         # Randomly decide whether to add, remove, or reverse an edge
#         if current_graph.has_edge(u, v):
#             current_graph.remove_edge(u, v)  # Remove the edge
#         else:
#             current_graph.add_edge(u, v)  # Add or reverse the edge
        
#         # Ensure the graph remains a DAG
#         if nx.is_directed_acyclic_graph(current_graph):
#             new_score = bayesian_score(current_graph, data)

#             # Accept the new graph if it's better or with a probability based on temperature
#             if new_score > current_score or random.random() < math.exp((new_score - current_score) / temp):
#                 current_score = new_score
#                 if new_score > best_score:
#                     best_score = new_score
#                     best_graph = current_graph.copy()
#             else:
#                 # Revert the change if not accepted
#                 if current_graph.has_edge(u, v):
#                     current_graph.remove_edge(u, v)
#                 else:
#                     current_graph.add_edge(u, v)

#         # Cool down the temperature
#         temp *= cooling_rate

#     return best_graph

# Function to save a .gph file from a NetworkX graph
def save_gph(G, file_path):
    with open(file_path, 'w') as f:
        for parent, child in G.edges():
            f.write(f'{parent},{child}\n')

# Function to parse the .gph file
def parse_gph(file_path):
    graph = nx.DiGraph()  # Directed graph for parent-child relationships
    with open(file_path, 'r') as file:
        for line in file:
            parent, child = line.strip().split(',')
            graph.add_edge(parent, child)  # Add directed edge from parent to child
    return graph

def visualize_gph(gph_file_path, dot_file_path, png_file_path):
    # Parse the .gph file
    G = parse_gph(gph_file_path)
    
    # Write the graph to a .dot file using NetworkX
    write_dot(G, dot_file_path)
    
    # Use GraphViz to convert the .dot file to a .png
    os.system(f"dot -Tpng {dot_file_path} -o {png_file_path}")
    print(f"Graph saved as {png_file_path}")

# def bayesian_score(G, data):
#     """
#     Computes the Bayesian score of the entire graph G given the data.
#     Args:
#         G: NetworkX DiGraph representing the Bayesian network structure.
#         data: pandas DataFrame with the observed data.
#     Returns:
#         The Bayesian score of the graph given the data.
#     """
#     total_score = 0

#     # Iterate over each node (variable) in the graph
#     for child in G.nodes():
#         ri = max(data[child])  # Number of unique states the child can take, determined by the maximum value
#         parents = list(G.predecessors(child))  # Get the parents of the child node
        
#         if len(parents) == 0:  # If no parents, compute based on marginal probability
#             # No parents, so calculate for marginal distribution
#             for k in range(1, ri + 1):  # Iterate over each possible state of the child
#                 m_ijk = len(data[data[child] == k])  # Count of how many times child is in state k
#                 total_score += gammaln(1 + m_ijk) - gammaln(1)  # Pseudocount = 1 for uniform prior
#             total_score += gammaln(ri) - gammaln(ri + len(data))
        
#         else:
#             # If there are parents, calculate conditional distribution
#             # Get all unique parent configurations
#             # parent_combinations = list(itertools.product(*[data[p].unique() for p in parents]))
#             parent_combinations = list(itertools.product(*[range(1, max(data[p]) + 1) for p in parents]))

#             for parent_inst in parent_combinations:
#                 # Filter data for this parent configuration
#                 subset = data.copy()
#                 for idx, parent in enumerate(parents):
#                     subset = subset[subset[parent] == parent_inst[idx]]
                
#                 # m_ij0: Total number of instances for this parent configuration
#                 m_ij0 = len(subset)
#                 total_score += gammaln(ri) - gammaln(ri + m_ij0)

#                 # Iterate over each possible state of the child
#                 for k in range(1, ri + 1):
#                     m_ijk = len(subset[subset[child] == k])  # Count of child being in state k
#                     total_score += gammaln(1 + m_ijk) - gammaln(1)  # Pseudocount = 1

#     return total_score

def bayesian_score(G, data, cache={}):
    """
    Computes the Bayesian score of the entire graph G given the data.
    Args:
        G: NetworkX DiGraph representing the Bayesian network structure.
        data: pandas DataFrame with the observed data.
        cache: Dictionary to store cached scores for node-parent configurations.
    Returns:
        The Bayesian score of the graph given the data.
    """
    total_score = 0

    # Convert DataFrame to NumPy array for efficient processing
    data_np = data.to_numpy()

    # Iterate over each node (variable) in the graph
    for child in G.nodes():
        parents = tuple(sorted(G.predecessors(child)))  # Get the parents of the child node, sorted to create a unique key
        
        # Create a unique cache key based on the child and its parents
        cache_key = (child, parents)

        if cache_key in cache:
            total_score += cache[cache_key]  # Use the cached score if available
        else:
            score = compute_local_bayesian_score_np(data, child, parents, data_np)
            cache[cache_key] = score  # Cache the score for future use
            total_score += score

    return total_score

def compute_local_bayesian_score_np(data, child, parents, data_np):
    """
    Computes the Bayesian score for a single node (child) and its parent set using NumPy for optimization.
    """
    total_score = 0
    child_idx = data.columns.get_loc(child)  # Get the index of the child in the DataFrame
    ri = np.max(data_np[:, child_idx])  # Number of unique states the child can take, determined by the maximum value

    if len(parents) == 0:  # If no parents, compute based on marginal probability
        # Vectorized computation for marginal distribution
        counts = np.bincount(data_np[:, child_idx], minlength=ri + 1)[1:]  # Counts for each state of the child (excluding 0)
        total_score += np.sum(gammaln(1 + counts) - gammaln(1))  # Pseudocount = 1
        total_score += gammaln(ri) - gammaln(ri + len(data_np))

    else:
        # Get the indices of the parents
        parent_idxs = [data.columns.get_loc(p) for p in parents]
        parent_combinations = np.array(list(itertools.product(*[range(1, np.max(data_np[:, idx]) + 1) for idx in parent_idxs])))

        for parent_inst in parent_combinations:
            # Vectorized filtering for parent configurations
            mask = np.all([data_np[:, parent_idxs[i]] == parent_inst[i] for i in range(len(parents))], axis=0)
            subset = data_np[mask]

            m_ij0 = len(subset)  # Total number of instances for this parent configuration
            total_score += gammaln(ri) - gammaln(ri + m_ij0)

            # Counts for each state of the child, given this parent configuration
            if len(subset) > 0:
                child_counts = np.bincount(subset[:, child_idx], minlength=ri + 1)[1:]  # Exclude 0
                total_score += np.sum(gammaln(1 + child_counts) - gammaln(1))

    return total_score

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]

    # Comment the following line to only visualize a gph file.
    compute(inputfilename, outputfilename)

    print("Bayesian score: ", bayesian_score(parse_gph(outputfilename), read_csv(inputfilename)))

    visualize_gph(outputfilename, outputfilename[:-3]+"dot", outputfilename[:-3]+"png")

if __name__ == '__main__':
    main()
