#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import functools
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch
from gn_block import GraphNetBlock
from utils.dataset import EncoderDecoderDataset


class MLP(nn.Module):
    def __init__(self, widths, act_fun=nn.ReLU, activate_final=None):
        super().__init__()

        layers = []

        n_in = widths[0]
        for i, w in enumerate(widths[1:-1]):
            linear = nn.Linear(n_in, w)
            layers.append(linear)

            act = act_fun()
            layers.append(act)

            n_in = w

        linear = nn.Linear(n_in, widths[-1])
        layers.append(linear)

        if activate_final is not None:
            act = activate_final()
            layers.append(act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MeshGraphNet(nn.Module):
    def __init__(self, output_size: int, latent_size: int, num_layers: int, n_nodefeatures: int,
                 n_edgefeatures_mesh: int,
                 n_edgefeatures_world: int,
                 message_passing_steps: int):
        """Encode-Process-Decode MeshGraphNet model."""
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
        network = MLP(widths, activate_final=None)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(output_size))
        return network

    def _encode_nodes(self, sample):
        node_features = sample['fluid'].node_attr
        env_features = sample['env'].node_attr
        N_fluid = node_features.shape[0]
        N_env = env_features.shape[0]

        combined_features = torch.cat([node_features, env_features], dim=0)
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

        env_edge_features = sample['env', 'wm_e', 'fluid'].edge_attr
        env_edge_latents = self.edgeset_encoders['world'](env_edge_features)
        sample['env', 'wm_e', 'fluid'].edge_attr = env_edge_latents
        
        return sample

    def _encode(self, sample: Batch) -> Batch:
        """Encodes node and edge features into latent features."""
        sample = self._encode_nodes(sample)
        sample = self._encode_edges(sample)
        return sample

    def _decode(self, sample):
        """Decodes node features from graph."""
        node_features = sample['fluid'].node_attr
        out_features = self.decoder(node_features)
        sample['fluid'].node_attr = out_features
        return sample

    def forward(self, sample) -> torch.Tensor:
        """Encodes and processes a multigraph, and returns node features."""
        sample = self._encode(sample)

        for i in range(self._message_passing_steps):
            sample = self.processor_steps[i](sample)

        return self._decode(sample)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
    output_size: int = 3
    latent_size: int = 64
    num_layers: int = 2
    n_nodefeatures: int = 3
    n_edgefeatures_mesh: int = 3
    n_edgefeatures_world: int = 3
    message_passing_steps: int = 15
    
    dataset = EncoderDecoderDataset()
    sample = dataset[0]

    enc_doc_model = MeshGraphNet(
        output_size=output_size,
        latent_size=latent_size,
        num_layers=num_layers,
        n_nodefeatures=n_nodefeatures,
        n_edgefeatures_mesh=n_edgefeatures_mesh,
        n_edgefeatures_world=n_edgefeatures_world,
        message_passing_steps=message_passing_steps).to('cuda')
    
    output = enc_doc_model(sample)
# %%
