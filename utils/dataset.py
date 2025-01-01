#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
import json

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump({k: v.tolist() if torch.is_tensor(v) else v for k, v in data.items()}, f)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {k: torch.tensor(v) if isinstance(v, list) else v for k, v in data.items()}

class VortexSheddingRe300To1000Dataset(Dataset):
    """PyTorch Geometric Dataset for Vortex Shedding simulation.
    
    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved
    split : str
        Dataset split ["train", "test"]
    transform : callable, optional
        Data transformation function
    pre_transform : callable, optional
        Data pre-transformation function
    """
    def __init__(self, root, split, transform=None, pre_transform=None):
        self.split = split
        super(VortexSheddingRe300To1000Dataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Usage example:
if __name__ == "__main__":
    # Create dataset
    dataset = VortexSheddingRe300To1000Dataset(
        root='../data',
        split='train'
    )
    
    # Access a single graph
    data = dataset[0]
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}")
    print(f"Sequence index: {data.sequence_idx.item()}")
    print(f"Time index: {data.time_idx.item()}")
# %%
