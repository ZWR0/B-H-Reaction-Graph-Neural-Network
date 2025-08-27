#!/usr/bin/env python
# coding: utf-8

# In[4]:

""" BRG: Buchwald-Hartwig Reaction Graph """

import torch
import torch_geometric.transforms as T
from tqdm import tqdm
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_names):
    """
    Preload and preprocess all data files.
    The first column in the file serves as the index for the DataFrame.
    
    """
    dfs = {}
    for name in file_names:
        scaler = StandardScaler()
        df = pd.read_csv(name, index_col=0)
        # Remove duplicate columns, columns with identical values, and columns containing NaN values
#         df = df.loc[:, ~df.T.duplicated()]
#         df = df.loc[:, (df.nunique() != 1)]
        df = df.dropna(axis=1, how='any')
#         df = pd.DataFrame(scaler.fit_transform(df))
        # Storing data with filenames (without extensions) as keys in a dictionary
        dfs[name.split('.')[0]] = df.values
    return dfs

def create_single_graph(features, edge_index):
    # All node feature arrays have the same length, and edges connect all nodes to form a complete graph.
    max_feat_dim = max(feature.shape[0] for feature in features)
    padded_features = torch.zeros(len(features), max_feat_dim)
    
    for i, feature in enumerate(features):
        padded_features[i, :feature.shape[0]] = torch.tensor(feature, dtype=torch.float)
        
    x = padded_features
    data = Data(x=x, edge_index=edge_index)
    data = T.NormalizeFeatures()(data)
    return data

def create_all_graphs(dfs, edge_index):
    datas = []
    for idx in tqdm(range(len(dfs[list(dfs.keys())[0]])), desc="Creating graphs"):
        reaction_features = [df[idx] for df in dfs.values()]
        data = create_single_graph(reaction_features, edge_index)
        datas.append(data)
    return datas

def visualize_graph_data(data, num_nodes, custom_positions=None, node_labels=None):
    """
    Visualize a directed graph from a PyTorch Geometric Data object.
    
    Args:
        data (Data): A PyTorch Geometric Data object representing a graph.
        custom_positions (dict): A dictionary with nodes as keys and positions as values.
        node_labels (dict): A dictionary with nodes as keys and labels as values.
        
    Returns:
        None: Displays the graph visualization using Matplotlib.
    """
    # Extract edge indices and convert to a list of (u, v) edge pairs
    edges = data.edge_index.t().tolist()
    
    # Create a directed NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))
    
    # Add edges to the graph; NetworkX's add_edges_from handles direction automatically
    G.add_edges_from(edges)
    
    # Visualize the graph
    plt.figure(figsize=(5, 4))  # Set figure size
    
    # Positioning the nodes using a layout algorithm or custom positions
    if custom_positions is None:
        pos = nx.spring_layout(G)
    else:
        pos = custom_positions
    
    # Draw the graph with customized node sizes, transparency, labels, and arrows for direction
    nx.draw_networkx(G, pos, node_size=700, alpha=1, with_labels=False, arrows=True, arrowstyle='->')
    
    # If node_labels is provided, draw the labels on the graph
    if node_labels is not None:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    
    plt.title("Directed Graph Visualization")
    plt.show()

class CustomGraphDataset(Dataset):
    def __init__(self, graphs, labels):
        """
        Initialize the dataset.
        
        Parameters:
            graphs: A list containing N `torch_geometric.data.Data` objects, each representing a graph.
            labels: A one-dimensional array or list of length N, containing the labels for each graph.
        """
        assert len(graphs) == len(labels), "The number of graphs must match the number of labels."
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        Retrieve the sample at a specified index.
        
        Parameters:
            idx: Index value.
            
        Returns:
            A tuple containing a `torch_geometric.data.Data` object and its corresponding label.
        """
        graph = self.graphs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return graph, label