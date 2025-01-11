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
from models.spatial_attention import SpatialTransformer
#IMPORTANT: Please only uncomment the following line if you are running this script independently
# from utils.dataset import EncoderDecoderDataset

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
        self.input_node_features_dim = input_node_features_dim      # 3
        self.input_edge_features_dim = input_edge_features_dim      # 3
        self.output_node_features_dim = output_node_features_dim    # 3
        self.internal_width = internal_width                        # 64
        self.message_passing_steps = message_passing_steps          # 32
        self.num_layers = num_layers                                # 2
        self.k = k
        self.PivotalNorm = torch.nn.LayerNorm(output_node_features_dim)
        
        self.num_heads = 4
        self.embedding_dim = 64
        self.position_dim = 2

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
        
        # get attention weights
        # self.spatial_attention = SpatialTransformer()
        
        self.feature_proj = nn.Linear(self.output_node_features_dim, self.embedding_dim)
        self.position_proj = nn.Linear(self.position_dim, self.embedding_dim)

        # Initialize MultiheadAttention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=self.num_heads, batch_first=True)
        
        # Initialize network weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(self.feature_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.feature_proj.bias, 0)

        nn.init.kaiming_normal_(self.position_proj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.position_proj.bias, 0)

        # Xavier initialization for MultiheadAttention weights
        for param in self.multihead_attn.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            


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
    
    def downsample(self, sample, position_mesh, position_pivotal, batch_size):
        new_sample = sample.clone()
        # Get total number of nodes
        node_features = new_sample['fluid'].node_attr
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
            x=node_features,                # [13592, 3]
            pos_x=position_mesh_batch,      # [13592, 2]
            pos_y=position_pivotal_batch,   # [2048, 2]
            batch_x=batch_mesh,             # [13592]
            batch_y=batch_pivotal,          # [2048]
        )

        # TODO: sptaial attention
        # node_features = self.spatial_attention(positions=position_pivotal_batch.float(), properties=node_features)
        query = self.feature_proj(node_features.reshape(-1, 256, 3)) + self.position_proj(position_pivotal_batch.float().reshape(-1, 256, 2))
        key = query
        value = query
        _, attn_weights = self.multihead_attn(query, key, value)
        node_features = (attn_weights @ node_features.reshape(-1, 256, 3)).reshape(-1, node_features.shape[-1])
        new_sample['fluid'].node_attr = node_features
        
        return new_sample
    
    def interpolate(self, sample, position_mesh, position_pivotal, batch_size):
        sample = sample.clone()
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
        
        return sample

    def encode(self, sample, position_mesh, position_pivotal, batch_size):
        sample = sample.clone()
        sample = self.encoder_processor(sample)
        node_features = self.PivotalNorm(sample['fluid'].node_attr)
        sample['fluid'].node_attr = node_features
        
        return sample

    def decode(self, sample, position_mesh, position_pivotal, batch_size):
        
        sample = self.decoder_processor(sample)
        return sample
    
    def residual_connection(self, out1, out2):
        
        out1['fluid'].node_attr += out2['fluid'].node_attr
        return out1
        
    
    def forward(self, sample, position_mesh, position_pivotal, batch_size):
        """Encodes and processes a multigraph, and returns node features."""
        out = sample
        out1 = self.encode(out, position_mesh, position_pivotal, batch_size)
        
        out2 = self.downsample(out1, position_mesh, position_pivotal, batch_size)
        out2 = self.interpolate(out2, position_mesh, position_pivotal, batch_size)
        
        # remove skip connection during the training of autoencoder, only use skip connection when training temporal attention
        # out = self.residual_connection(out1, out2)
        # use when training autoencoder
        out = out2

        return self.decode(out, position_mesh, position_pivotal, batch_size)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
    import numpy as np
    from utils import constant

    C = constant.Constant()
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
