import os
import warnings
import math

from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, models
from torchdrug.layers import functional
from torchdrug.core import Registry as R 

from torch_scatter import scatter_add
from torchdrug.layers.functional import variadic_to_padded

from NCprior import layer
import ipdb


@R.register("models.NCPriorClassifier")
class NCPriorClassifier(nn.Module, core.Configurable):

    def __init__(self, in_channels):
        super(NCPriorClassifier, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels,
                                kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        modules = [layer.ResidualBlockA(in_channels), 
                   layer.ResidualBlockA(in_channels),
                   layer.ResidualBlockA(in_channels),
                   layer.ResidualBlockB(in_channels),
                   layer.ResidualBlockA(in_channels*2),
                   layer.ResidualBlockA(in_channels*2),
                   layer.ResidualBlockA(in_channels*2),
                   layer.ResidualBlockB(in_channels*2)]
        self.residual_blocks = nn.Sequential(*modules)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Linear(in_channels * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #xx = x.clone()
        x = self.conv(x)
        x = self.relu(x)
        x = self.residual_blocks(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        #ipdb.set_trace()

        return x

@R.register("models.VAE")
class VAE(nn.Module, core.Configurable):

    def __init__(self, 
            in_channels: int,
            image_rows: int,
            image_cols: int,
            latent_dim: int,
            num_layer: int,
            hidden_dims: List,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1
        ):
                
        super(VAE, self).__init__()

        if len(hidden_dims) == 1:
            d = hidden_dims[0]
            hidden_dims = [d * (2 ** i) for i in range(num_layer)]
        if num_layer is None:
            num_layer = len(hidden_dims)
        
        self.in_channels = in_channels
        self.image_rows = image_rows
        self.image_cols = image_cols

        self.latent_dim = latent_dim
        self.num_layer = num_layer
        self.hidden_dims = hidden_dims
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        row_dim = self.image_rows
        for i in range(self.num_layer):
            row_dim = math.floor((row_dim - self.kernel_size + 2 * self.padding) / self.stride) + 1
        col_dim = self.image_cols
        for i in range(self.num_layer):
            col_dim = math.floor((col_dim - self.kernel_size + 2 * self.padding) / self.stride) + 1
        self.flatten_dim = self.hidden_dims[-1] * row_dim * col_dim
        self.flatten_shape = (self.hidden_dims[-1], row_dim, col_dim)

        
        self.encoder = self._construct_encoder([self.in_channels] + self.hidden_dims)

        self.mu_mlp = nn.Linear(self.flatten_dim, self.latent_dim) # NOTE
        self.var_mlp = nn.Linear(self.flatten_dim, self.latent_dim)

        self.decoder_mlp = nn.Linear(self.latent_dim, self.flatten_dim)
        self.decoder = self._construct_decoder(self.hidden_dims[::-1] + [self.hidden_dims[0]])

        self.final_layer = nn.Sequential(
                                nn.Conv2d(self.hidden_dims[0], self.in_channels,
                                          kernel_size=self.kernel_size,
                                          padding=self.padding),
                                nn.Sigmoid()
                            )

    
    def _construct_decoder(self, hidden_dims):

        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i+1],
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        return nn.Sequential(*modules)


    def _construct_encoder(self, hidden_dims):

        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1],
                             kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        return nn.Sequential(*modules)
    
    def encode(self, input):
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1) # (bsz, 512)
        mu = self.mu_mlp(x)
        logvar = self.var_mlp(x)

        return [mu, logvar]
    
    def decode(self, z):
        x = self.decoder_mlp(z)
        x = x.view(-1, *self.flatten_shape)
        x = self.decoder(x)
        x = self.final_layer(x)

        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar # (bsz, 1, 32, 32), (bsz, latent_dim), (bsz, latent_dim)

    def sample(self, num_samples):

        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        samples = self.decode(z)

        return samples
