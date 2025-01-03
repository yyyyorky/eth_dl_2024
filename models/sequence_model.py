#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import torch.nn as nn
from utils.constant import Constant
from models.autoencoder import MeshReduce
from models.decoder_layer import DecoderLayer, TransformerDecoder

C = Constant()

#for testing
from utils.dataset import TemporalSequenceLatentDataset
from torch.utils.data import DataLoader
from models.mlp import MLP


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class SequenceModel(nn.Module):
    '''
    Decoder-only multi-head attention architecture
    '''
    def __init__(
        self,
        input_dim: int,
        input_context_dim: int,
        dropout_rate: float = 0,
        num_layers_decoder: int = 3,
        num_heads: int = 8,
        dim_feedforward_scale: int = 4,
        num_layers_context_encoder: int = 2,
        num_layers_input_encoder: int = 2,
        num_layers_output_encoder: int = 2,
    ):
        super(SequenceModel, self).__init__()

        self.input_dim = input_dim
        self.input_context_dim = input_context_dim
        self.dropout_rate = dropout_rate
        self.num_layers_decoder = num_layers_decoder
        self.num_heads = num_heads
        self.dim_feedforward_scale = dim_feedforward_scale
        self.num_layers_context_encoder = num_layers_context_encoder
        self.num_layers_input_encoder = num_layers_input_encoder
        self.num_layers_output_encoder = num_layers_output_encoder

        decoder_norm =nn.LayerNorm(input_dim)
        self.decoder_temporal = TransformerDecoder(
            DecoderLayer,
            num_layers=self.num_layers_decoder,
            norm=decoder_norm,
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * dim_feedforward_scale,
            dropout=dropout_rate,
        )

        self.input_encoder = self._make_mlp(input_dim, input_dim, num_layers=num_layers_input_encoder)
        self.output_encoder = self._make_mlp(input_dim, input_dim, num_layers=num_layers_output_encoder)
        self.context_encoder = self._make_mlp(input_context_dim, input_dim, num_layers=num_layers_context_encoder)

    def _make_mlp(self, input_size: int, output_size: int, layer_norm: bool = True, num_layers = 2) -> nn.Module:
        widths = [input_size] + [input_size * 2] * num_layers + [output_size]
        network = MLP(widths, activate_final=None)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(output_size))
        return network
    
    @torch.no_grad()
    def sample(self, z0, step_size, context=None):
        """
        Samples a sequence starting from the initial input `z0` for a given number of steps using
        the model's `forward` method.
        """
        z = z0  # .unsqueeze(1)

        for i in range(step_size):
            prediction = self.forward(z, context)[:, -1].unsqueeze(1)
            z = torch.concat([z, prediction], dim=1)
        return z
    
    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = C.device,
        dtype: torch.dtype = torch.get_default_dtype(),
    ) :
        """Generates a square mask for the sequence. The mask shows which entries should not be used."""

        return torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
    
    def forward(self, x, context=None):
        '''
        input time series data at t = [0, 1, 2, ...]
        output time series data at t = [1, 2, 3, ...]
        Args:
            x: Input tensor of shape (batch_size, seq_len, nodes_num, nodes_features)
            context: Input tensor of shape (batch_size, context_dim)
        '''
        if len(x.shape) != 4:
            raise ValueError("please only input batched data")
        batch_size, seq_len, nodes_num, nodes_features = x.shape
        x = x.reshape(batch_size, seq_len, self.input_dim)
        #x: [batch_size, seq_len, input_dim]

        if context is not None:
            context = self.context_encoder(context)
            context = context.unsqueeze(1)
            x = torch.cat([context, x], dim=1)
        x = self.input_encoder(x)
        mask = self.generate_square_subsequent_mask(
            x.size()[1], device=C.device
        )
        output = self.decoder_temporal(x, mask=mask)
        output = self.output_encoder(output)
        output = output[:, 1:]
        output = output.reshape(batch_size, -1, nodes_num, nodes_features)
        return output






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':

    position_mesh = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_all.txt"))).to(C.device)
    position_pivotal = torch.from_numpy(np.loadtxt(os.path.join(C.data_dir, "meshPosition_pivotal.txt"))).to(C.device)
    tsl_dataset = TemporalSequenceLatentDataset( 
                                                split='train', 
                                                position_mesh=position_mesh, 
                                                position_pivotal=position_pivotal,
                                                produce_latent=False)
    tsl_loader = DataLoader(tsl_dataset, batch_size=1, shuffle=False)
    model = SequenceModel(
        input_dim=768,
        input_context_dim= 1,
        num_layers_decoder=3,
        num_heads=8,
        dim_feedforward_scale=4,
        num_layers_context_encoder=2,
        num_layers_input_encoder=2,
        num_layers_output_encoder=2,
    ).to(C.device)
    for x, context in tsl_loader:
        x = x[:, :2]
        print(x.shape)
        out = model(x, context)
        print(out.shape)
        break
    


# %%
