import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset

class TransformerModel(nn.Module):

        def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                        nlayers: int, dropout: float = 0.5):
                super().__init__()
                self.model_type = 'Transformer'
                #self.pos_encoder = PositionalEncoding(d_model, dropout)

                #Multi-headed self attention layer
                encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
                self.encoder = nn.Embedding(ntoken, d_model)
                self.d_model = d_model
                self.decoder = nn.Linear(d_model, ntoken)

                #Weight Initialization
                initrange = 0.1
                self.encoder.weight.data.uniform_(-initrange, initrange)
                self.decoder.bias.data.zero_()
                self.decoder.weight.data.uniform_(-initrange, initrange)

        def call(self, src: Tensor, src_mask: Tensor):
                "Forward pass"

                src = self.encoder(src) * math.sqrt(self.d_model)
                src = self.pos_encoder(src)
                output = self.transformer_encoder(src, src_mask)
                output = self.decoder(output)
                return output
