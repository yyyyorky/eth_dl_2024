#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import numpy as np
from gn_block import GraphNetBlock
from torch import nn
import mlp
import functools
from torch_geometric.data import Batch

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class MeshGraphNet(nn.Module):
    def __init__(self, output_size: int, latent_size: int, num_layers: int, n_nodefeatures: int,
                 n_edgefeatures_mesh: int,
                 n_edgefeatures_world: int,
                 message_passing_steps: int):
        """Encode-Process-Decode GraphNet model."""
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self.n_nodefeatures = n_nodefeatures
        self.n_edgefeatures_mesh = n_edgefeatures_mesh
        self.n_edgefeatures_world = n_edgefeatures_world
        self._message_passing_steps = message_passing_steps

        self.node_encoder = self._make_mlp(self.n_nodefeatures, self._latent_size)
        self.decoder = self._make_mlp(self._latent_size, self._output_size, layer_norm=False)

        edgeset_encoders = {}
        edgeset_encoders['mesh'] = self._make_mlp(self.n_edgefeatures_mesh, self._latent_size)
        edgeset_encoders['world'] = self._make_mlp(self.n_edgefeatures_world, self._latent_size)
        self.edgeset_encoders = nn.ModuleDict(edgeset_encoders)

        node_proc_model = functools.partial(self._make_mlp, input_size=self._latent_size * (1 + 2),
                                            output_size=self._latent_size)
        edge_proc_model = functools.partial(self._make_mlp, input_size=self._latent_size * 3,
                                            output_size=self._latent_size)

        processor_steps = []
        for i in range(message_passing_steps):
            processor_steps.append(GraphNetBlock(node_proc_model, edge_proc_model))
        self.processor_steps = nn.ModuleList(processor_steps)
        self.i = 0

    def _make_mlp(self, input_size: int, output_size: int, layer_norm: bool = True) -> nn.Module:
        """Builds an MLP."""
        widths = [input_size] + [self._latent_size] * self._num_layers + [output_size]
        network = mlp.MLP(widths, activate_final=None)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(output_size))
        return network

    def _encode_nodes(self, sample):
        fluid_features = sample['fluid'].node_attr
        env_features = sample['env'].node_attr
        N_fluid = fluid_features.shape[0]
        N_env = env_features.shape[0]

        combined_features = torch.cat([fluid_features, env_features], dim=0)
        combined_latents = self.node_encoder(combined_features)

        fluid_latents = combined_latents[:N_fluid]
        env_latents = combined_latents[N_fluid:]

        sample['fluid'].node_attr = fluid_latents
        sample['env'].node_attr = env_latents

        return sample

    def _encode_edges(self, sample):
        mesh_edge_features = sample['fluid', 'm_e', 'fluid'].edge_attr
        mesh_edge_latents = self.edgeset_encoders['mesh'](mesh_edge_features)
        sample['fluid', 'm_e', 'fluid'].edge_attr = mesh_edge_latents

        env_fluid_edge_features = sample['env', 'wm_e', 'fluid'].edge_attr
        N_world_edges = env_fluid_edge_features.shape[0]

        

    def _encode(self, sample: Batch) -> Batch:
        """Encodes node and edge features into latent features."""
        sample = self._encode_nodes(sample)
        sample = self._encode_edges(sample)
        return sample

    def _decode(self, sample):
        """Decodes node features from graph."""
        cloth_features = sample['cloth'].node_features
        out_features = self.decoder(cloth_features)
        sample['cloth'].node_features = out_features
        return sample

    def forward(self, sample) -> torch.Tensor:
        """Encodes and processes a multigraph, and returns node features."""
        sample = self._encode(sample)

        for i in range(self._message_passing_steps):
            sample = self.processor_steps[i](sample)

        return self._decode(sample)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
    pass