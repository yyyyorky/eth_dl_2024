#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Dataset, HeteroData
import json
import constant

C = constant.Constant()

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump({k: v.tolist() if torch.is_tensor(v) else v for k, v in data.items()}, f)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {k: torch.tensor(v) if isinstance(v, list) else v for k, v in data.items()}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
The node features include (3 in total):

- Velocity components at time step t, i.e., u_t, v_t
- Pressure at time step t, p_t

The edge features for each sample are time-independent and include (3 in total):

- Relative x and y distance between the two end nodes of an edge
- L2 norm of the relative distance vector

The output of the model is the velocity components for the following steps, i.e.,
u_{t+1}, v_{t+1}, p_{t+1}.
'''

class EncoderDecoderDataset(Dataset):
    """PyTorch Geometric Dataset for Graph Autoencoder
    Parameters
    ----------
    data_dir : str
        The directory of the data
    split : str
        Dataset split ["train", "test"]
    """
    def __init__(self, data_dir = C.data_dir, split='train', device=C.device):
        self.split = split
        super(EncoderDecoderDataset, self).__init__()

        self.device = device
        self.data_dir = data_dir

        self.rawData = np.load(
            os.path.join(self.data_dir, "rawData.npy"), allow_pickle=True
        )

        # select training and testing set
        if self.split == "train":
            self.sequence_ids = [i for i in range(101) if i % 2 == 0]
        if self.split == "test":
            self.sequence_ids = [i for i in range(101) if i % 2 == 1]

        # solution states
        self.solution_states = torch.from_numpy(
            self.rawData["x"][self.sequence_ids, :, :, :]
        ).float().to(self.device)

        # edge information
        self.edge_attr = torch.from_numpy(
            self.rawData["edge_attr"]
        ).float().to(self.device)
        
        # edge connection
        self.edge_index = torch.from_numpy(
            self.rawData["edge_index"]
        ).type(torch.long).to(self.device)

        # sequence length info
        self.sequence_len = self.solution_states.shape[1]
        self.sequence_num = self.solution_states.shape[0]
        self.num_nodes = self.solution_states.shape[2]

        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            self.edge_stats = load_json(self.data_dir + "/edge_stats.json")
        self.edge_stats["edge_mean"] = self.edge_stats["edge_mean"].to(self.device)
        self.edge_stats["edge_std"] = self.edge_stats["edge_std"].to(self.device)

        if self.split == "train":
            self.node_stats = self._get_node_stats()
        else:
            self.node_stats = load_json(self.data_dir + "/node_stats.json")
        self.node_stats["node_mean"] = self.node_stats["node_mean"].to(self.device)
        self.node_stats["node_std"] = self.node_stats["node_std"].to(self.device)

        # handle normalization
        for i in range(self.sequence_num):
            for j in range(self.sequence_len):
                self.solution_states[i, j] = self.normalize(
                    self.solution_states[i, j],
                    self.node_stats["node_mean"],
                    self.node_stats["node_std"],
                )
        self.edge_attr = self.normalize(
            self.edge_attr, self.edge_stats["edge_mean"], self.edge_stats["edge_std"]
        )

        self.re = self.get_re_number()

    def get_re_number(self):
        """Get RE number"""
        ReAll = torch.from_numpy(np.linspace(300, 1000, 101)).float().reshape([-1, 1]).to(self.device)
        ReAll = ReAll/ReAll.max()
        if self.split == "train":
            index = [i for i in range(101) if i % 2 == 0]
        else:
            index = [i for i in range(101) if i % 2 == 1]
        ReAll = ReAll[index]
        ReAll = ReAll.repeat(self.sequence_len, 1)
        return ReAll

        

    def len(self):
        return self.sequence_num * self.sequence_len

    def get(self, idx):
        '''
        Get the data object for the idx-th sequence in the dataset.
        'x': node features [num_nodes, num_node_features]
        'edge_index': edge connections [2, num_edges]
        'edge_attr': edge features [num_edges, num_edge_features]
        'y': node targets [num_nodes, num_node_features]
        'sequence_idx': sequence index [1]
        'time_idx': time index [1]
        'global_features': global features [1, num_global_features] (Re, Nu)
        '''
        sidx = idx // self.sequence_len
        tidx = idx % self.sequence_len

        # #######################################################################
        # node_features = self.solution_states[sidx, tidx]
        # node_targets = self.solution_states[sidx, tidx]
        # # node_targets = node_features because it is an autoencoder
        # re = self.re[sidx]

        # data = Data(
        #     x=node_features,
        #     edge_index=self.edge_index,
        #     edge_attr=self.edge_attr,
        #     y=node_targets,
        #     sequence_idx=torch.tensor([sidx], device=self.device),
        #     time_idx=torch.tensor([tidx], device=self.device),
        #     re = re,
        # )
        # #######################################################################

        data = HeteroData()
        node_features = self.solution_states[sidx, tidx]
        node_targets = self.solution_states[sidx, tidx]
        data['fluid'].node_attr = node_features
        data['fluid'].node_target = node_targets
  
        # Environment node (Re)
        re = self.re[sidx]
        re = re.repeat(1, 3)  # Shape: [1, 3]
        data['env'].node_attr = re
        
        # Original fluid-to-fluid edges
        data['fluid', 'm_e', 'fluid'].edge_index = self.edge_index
        data['fluid', 'm_e', 'fluid'].edge_attr = self.edge_attr
        
        # Environment-to-fluid edges: connect Re node to all fluid nodes
        # the first is sender nodes, the second is receiver nodes
        num_fluid_nodes = node_features.size(0)
        env_to_fluid_edge_index = torch.zeros((2, num_fluid_nodes), device=self.device, dtype=torch.long)
        env_to_fluid_edge_index[1] = torch.arange(num_fluid_nodes, device=self.device)  # Target nodes
        data['env', 'wm_e', 'fluid'].edge_index = env_to_fluid_edge_index
        data['env', 'wm_e', 'fluid'].edge_attr = torch.ones((num_fluid_nodes, self.edge_attr.size(1)), device=self.device)

        data['fluid'].sequence_idx = torch.tensor([sidx], device=self.device)
        data['fluid'].time_idx = torch.tensor([tidx], device=self.device)

        return data

    def _get_edge_stats(self):
        stats = {
            "edge_mean": self.edge_attr.mean(dim=0),
            "edge_std": self.edge_attr.std(dim=0),
        }
        save_json(stats, self.data_dir + "/edge_stats.json")
        return stats

    def _get_node_stats(self):
        stats = {
            "node_mean": self.solution_states.mean(dim=[0, 1, 2]),
            "node_std": self.solution_states.std(dim=[0, 1, 2]),
        }
        save_json(stats, self.data_dir + "/node_stats.json")
        return stats

    @staticmethod
    def normalize(invar, mu, std):
        if invar.size()[-1] != mu.size()[-1] or invar.size()[-1] != std.size()[-1]:
            raise ValueError(
                "invar, mu, and std must have the same size in the last dimension"
            )
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def denormalize(invar, mu, std):
        return invar * std + mu
    

class TemporalSequenceLatentDataset(Dataset):
    '''PyTorch Geometric Dataset for Temporal Latent Sequence
        This is for prediction of latent token temporal sequences.
        The sequence is a time series of latent tokens.
        For the time series of graphs, use the TemporalSequenceGraphDataset.
    '''
    pass

    

class TemporalSequenceGraphDataset(Dataset):
    """PyTorch Geometric Dataset for Temporal Graph Sequence
        This is for end-to-end prediction of temporal sequences.
        The sequence is a time series of graphs.
        For the time series of latent tokens, use the TemporalSequenceLatentDataset.
    Parameters
    ----------
    data_dir : str
        The directory of the data
    split : str
        Dataset split ["train", "test"]
    """
    pass

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Usage example:
if __name__ == "__main__":
    # Create dataset
    dataset = EncoderDecoderDataset()
    print(dataset[0].keys)
    print(dataset[0].node_types)
    print(dataset[0].edge_types)
# %%
