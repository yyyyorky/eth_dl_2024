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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



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

        decoder_norm =nn.LayerNorm(input_dim, eps=1e-5, bias=True)
        self.decoder = TransformerDecoder(
            DecoderLayer,
            num_layers=self.num_layers_decoder,
            norm=decoder_norm,
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * dim_feedforward_scale,
            dropout=dropout_rate,
        )




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == '__main__':
    pass