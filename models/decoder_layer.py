#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from typing import  Optional

from torch import nn
from torch import Tensor
from typing import Optional, Union, Type


class TransformerDecoder(nn.Module):
    """A stack of N decoder layers.
    
    Parameters:
        decoder_layer: Decoder layer class or instance to be stacked
        num_layers: Number of decoder layers to stack
        norm: Optional normalization layer
        **kwargs: Additional arguments to pass to decoder_layer if it's a class
    """
    def __init__(
        self, 
        decoder_layer: Type[nn.Module], 
        num_layers: int, 
        norm: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__()
        
        # # If decoder_layer is a class, instantiate it with kwargs
        # if isinstance(decoder_layer, type):
        #     decoder_layer = decoder_layer(**kwargs)
            
        # Create num_layers copies of the decoder layer
        self.layers = nn.ModuleList([decoder_layer(**kwargs) for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor
            mask: Optional attention mask
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        if self.norm is not None:
            x = self.norm(x)
            
        return x

class DecoderLayer(nn.Module):
    """A decoder-only transformer layer.
    
    Parameters:
        d_model: Dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of the feedforward network
        dropout: Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def feed_forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        """
        attn_out = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
