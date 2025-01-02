#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
import torch_scatter
from models.mgn import MeshGraphNet
from utils.dataset import EncoderDecoderDataset

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class MeshReduce(nn.Module):
    def __init__(self, 
                 input_node_features_dim: int,
                 input_edge_features_dim: int,
                 output_node_features_dim: int,
                 internal_width: int,
                 message_passing_steps: int,
                 num_layers: int,
                 k: int = 3
                 ):
        super(MeshReduce, self).__init__()
        self.input_node_features_dim = input_node_features_dim
        self.input_edge_features_dim = input_edge_features_dim
        self.output_node_features_dim = output_node_features_dim
        self.internal_width = internal_width
        self.message_passing_steps = message_passing_steps
        self.num_layers = num_layers
        self.k = k
        self.PivotalNorm = torch.nn.LayerNorm(output_node_features_dim)

        self.encoder_processor = MeshGraphNet(
            output_size=output_node_features_dim,
            latent_size=internal_width,
            num_layers=num_layers,
            n_nodefeatures=input_node_features_dim,
            n_edgefeatures_mesh=input_edge_features_dim,
            n_edgefeatures_world=input_edge_features_dim,
            message_passing_steps=message_passing_steps//2
        )

        self.decoder_processor = MeshGraphNet(
            output_size=input_node_features_dim,
            latent_size=internal_width,
            num_layers=num_layers,
            n_nodefeatures=output_node_features_dim,
            n_edgefeatures_mesh=output_node_features_dim,
            n_edgefeatures_world=output_node_features_dim,
            message_passing_steps=message_passing_steps//2
        )

    def knn_interpolate(
        self,
        x: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        batch_x: torch.Tensor = None,
        batch_y: torch.Tensor = None,
        k: int = 3,
        num_workers: int = 4,
    ):
        with torch.no_grad():
            assign_index = torch_cluster.knn(
                pos_x,
                pos_y,
                k,
                batch_x=batch_x,
                batch_y=batch_y,
                num_workers=num_workers,
            )
            y_idx, x_idx = assign_index[0], assign_index[1]
            diff = pos_x[x_idx] - pos_y[y_idx]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        y = torch_scatter.scatter(
            x[x_idx] * weights, y_idx, 0, dim_size=pos_y.size(0), reduce="sum"
        )
        y = y / torch_scatter.scatter(
            weights, y_idx, 0, dim_size=pos_y.size(0), reduce="sum"
        )

        return y.float(), x_idx, y_idx, weights

    def encode(self, sample, position_mesh, position_pivotal, batch_size):
        sample = self.encoder_processor(sample)
        node_features = self.PivotalNorm(sample['fluid'].node_attr)
        sample['fluid'].node_attr = node_features
        
        # Get total number of nodes
        total_nodes = node_features.shape[0]
        nodes_per_batch = total_nodes // batch_size
        
        # Handle the last batch which might have remaining nodes
        remaining_nodes = total_nodes % batch_size
        
        # Create batch indices accounting for uneven last batch
        nodes_index = torch.arange(batch_size).to(node_features.device)
        if remaining_nodes == 0:
            batch_mesh = torch.repeat_interleave(nodes_index, nodes_per_batch)
        else:
            # Create list of nodes per batch with last batch having the remainder
            nodes_per_batch_list = [nodes_per_batch] * (batch_size - 1) + [nodes_per_batch + remaining_nodes]
            batch_mesh = torch.repeat_interleave(nodes_index, torch.tensor(nodes_per_batch_list).to(node_features.device))
        
        # Create pivotal batch indices
        pivotal_nodes_per_batch = len(position_pivotal)
        batch_pivotal = torch.repeat_interleave(nodes_index, pivotal_nodes_per_batch)
        
        # Create position tensors
        position_mesh_batch = position_mesh.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, position_mesh.shape[-1])
        position_pivotal_batch = position_pivotal.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, position_pivotal.shape[-1])

        node_features, _, _, _ = self.knn_interpolate(
            x=node_features,
            pos_x=position_mesh_batch,
            pos_y=position_pivotal_batch,
            batch_x=batch_mesh,
            batch_y=batch_pivotal,
        )
        
        sample['fluid'].node_attr = node_features
        return sample

    def decode(self, sample, position_mesh, position_pivotal, batch_size):
        node_features = sample['fluid'].node_attr
        
        # Get total number of nodes
        total_nodes = node_features.shape[0]
        pivotal_nodes_per_batch = total_nodes // batch_size
        mesh_nodes_per_batch = len(position_mesh)
        
        # Handle the last batch which might have remaining nodes
        remaining_nodes = total_nodes % batch_size
        
        # Create batch indices
        nodes_index = torch.arange(batch_size).to(node_features.device)
        batch_mesh = torch.repeat_interleave(nodes_index, mesh_nodes_per_batch)
        
        if remaining_nodes == 0:
            batch_pivotal = torch.repeat_interleave(nodes_index, pivotal_nodes_per_batch)
        else:
            # Create list of nodes per batch with last batch having the remainder
            nodes_per_batch_list = [pivotal_nodes_per_batch] * (batch_size - 1) + [pivotal_nodes_per_batch + remaining_nodes]
            batch_pivotal = torch.repeat_interleave(nodes_index, torch.tensor(nodes_per_batch_list).to(node_features.device))
        
        # Create position tensors
        position_mesh_batch = position_mesh.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, position_mesh.shape[-1])
        position_pivotal_batch = position_pivotal.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, position_pivotal.shape[-1])

        node_features, _, _, _ = self.knn_interpolate(
            x=node_features,
            pos_x=position_pivotal_batch,
            pos_y=position_mesh_batch,
            batch_x=batch_pivotal,
            batch_y=batch_mesh,
        )
        
        sample['fluid'].node_attr = node_features
        sample = self.decoder_processor(sample)
        return sample
    
    def forward(self, sample, position_mesh, position_pivotal, batch_size):
        """Encodes and processes a multigraph, and returns node features."""
        sample = self.encode(sample, position_mesh, position_pivotal, batch_size)

        return self.decode(sample, position_mesh, position_pivotal, batch_size)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
from utils import constant

C = constant.Constant()

if __name__ == '__main__':
    # Fix the bug: output_node_features_dim: 3 -> 64
    output_node_features_dim: int = 3
    internal_width: int = 64
    num_layers: int = 2
    input_node_features_dim: int = 3
    input_edge_features_dim: int = 3
    message_passing_steps: int = 16
    
    dataset = EncoderDecoderDataset()
    sample = dataset[0]

    enc_doc_model = MeshReduce(input_node_features_dim,
                                input_edge_features_dim,
                                output_node_features_dim,
                                internal_width,
                                message_passing_steps,
                                num_layers).to('cuda')
    
    position_mesh = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_all.txt"))).to("cuda")
    position_pivotal = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_pivotal.txt"))).to("cuda")
    
    encode_output = enc_doc_model.encode(sample, position_mesh, position_pivotal, 1)
    
    print(encode_output)
    
    decode_output = enc_doc_model.decode(encode_output, position_mesh, position_pivotal, 1)
    
    print(decode_output)

    trial = enc_doc_model(sample, position_mesh, position_pivotal, 1)
# %%
