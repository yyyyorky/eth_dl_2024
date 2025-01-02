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
            n_edgefeatures_mesh=internal_width,
            n_edgefeatures_world=internal_width,
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
        
        nodes_index = torch.arange(batch_size).to(node_features.device)
        batch_mesh = nodes_index.repeat_interleave(node_features.shape[0])
        position_mesh_batch = position_mesh.repeat(batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(
            torch.tensor([len(position_pivotal)] * batch_size).to(node_features.device)
        )

        node_features, _, _, _ = self.knn_interpolate(
                                        x=node_features,
                                        pos_x=position_mesh_batch,
                                        pos_y=position_pivotal_batch,
                                        batch_x=batch_mesh,
                                        batch_y=batch_pivotal,
        )
        
        # print(node_features.shape)
        
        # print(position_mesh_batch.shape)
        
        # print(position_pivotal_batch.shape)
        
        # print(batch_mesh.shape)
        
        # print(batch_pivotal.shape)
        
        sample['fluid'].node_attr = node_features

        #TODO: add the spatial information embedding to the pivotal nodes.
        #      add spatial attention only to the pivotal nodes.
        #      Here is it better to return a graph or only the tokenized nodes?

        return sample
    

    def decode(self, sample, position_mesh, position_pivotal, batch_size):
        node_features = sample['fluid'].node_attr

        nodes_index = torch.arange(batch_size).to(node_features.device)
        # Fix the bug: node_features -> position_mesh
        batch_mesh = nodes_index.repeat_interleave(position_mesh.shape[0])
        position_mesh_batch = position_mesh.repeat(batch_size, 1)
        position_pivotal_batch = position_pivotal.repeat(batch_size, 1)
        batch_pivotal = nodes_index.repeat_interleave(
            torch.tensor([len(position_pivotal)] * batch_size).to(node_features.device)
        )
        
        print(node_features.shape)
        
        # print(position_mesh_batch.shape)
        
        # print(position_pivotal_batch.shape)
        
        # print(batch_mesh.shape)
        
        # print(batch_pivotal.shape)

        node_features, _, _, _ = self.knn_interpolate(
                                        x=node_features,
                                        pos_x=position_pivotal_batch,
                                        pos_y=position_mesh_batch,
                                        batch_x=batch_pivotal,
                                        batch_y=batch_mesh,
        )
        sample['fluid'].node_attr = node_features
        
        print(node_features.shape)

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
    output_node_features_dim: int = 64
    internal_width: int = 64
    num_layers: int = 2
    input_node_features_dim: int = 3
    input_edge_features_dim: int = 3
    message_passing_steps: int = 15
    
    dataset = EncoderDecoderDataset()
    sample = dataset[0]
    
    print(sample)

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
# %%
