from typing import Tuple
import torch
from torch import nn
import numpy as np

from config import Config
from channel import Channel
from shrink import Shrink
from loss import Loss

class Tracker:
    def __init__(self, x, y, U, s, Vh, sigma2):
        pass
        

class VAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """
        VAMP iteration (layer)

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.shrinkage = Shrink(config, 'bayes')
        self.tol = [1e-20, -1e-20]
        
    def forward(self, 
                ytilde: torch.Tensor, 
                r: torch.Tensor, 
                g: torch.Tensor,
                channel: Channel,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            y (torch.Tensor): _description_
            r (torch.Tensor): _description_
            g (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        xamp , var = self.shrinkage(r, g)
        a = g * var.mean(dim=1).unsqueeze(-1)
        rtilde = (xamp - a * r) / self.regularize(1 - a)
        gtilde = g * (1 - a) / self.regularize(a)
        t = channel.sigma2 * channel.s2 + gtilde.squeeze(-1).expand(-1, channel.R)
        d = (channel.sigma2 * channel.s2 / self.regularize(t)).unsqueeze(-1)
        dmean = d.mean(dim=1).unsqueeze(-1)
        g = gtilde * dmean / self.regularize(channel.M / channel.R - dmean)
        z = (d / dmean) * (ytilde - channel.Vh @ rtilde)
        r = rtilde +  channel.M / channel.R * channel.V @ z

        return xamp, r, g
    
    def regularize(self, a:torch.Tensor):
        a[a==0] = np.random.choice(self.tol)
        return a

class VAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        """
        VAMP model

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.device = config.device
        self.B = config.B
        self.sparsity = config.sparsity
        self.N_Layers = config.N_Layers
        
        # setup
        self.layers = nn.ModuleList([VAMPLayer(config) for _ in range(self.N_Layers+1)])
        self.loss = Loss(config)
        
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor, 
                U: torch.Tensor,
                s: torch.Tensor,
                Vh: torch.Tensor,
                SNR: float,
                symbols,
                indices,
                ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            torch.Tensor: _description_
        """
        self.loss.init()
        channel.generate_svd()
        ytilde = torch.diag(1/channel.s) @ (channel.Uh @ y)
        r = torch.zeros(self.xsize, dtype=self.datatype, requires_grad=False, device=self.device)
        g = self.sparsity
        for i, layer in enumerate(self.layers):
            xamp, r, g = layer(ytilde, r, g, channel)
            if i > 1:
                self.loss(xamp, x)
        return self.loss
         
if __name__ == "__main__":
    pass