from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from loss import Loss
from channel import Channel
from shrink import Shrink


class BAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.shrinkage = Shrink(config, "bayes")
        
    def forward(self,
                y: torch.Tensor,
                xamp_prev: torch.Tensor,
                var_prev: torch.Tensor,
                V_prev: torch.Tensor,
                Z_prev: torch.Tensor,
                channel: Channel
                )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            y (torch.Tensor): _description_
            xamp (torch.Tensor): _description_
            var (torch.Tensor): _description_
            V (torch.Tensor): _description_
            Z (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        V_next = channel.Habs2 @ var_prev
        Z_next = channel.H @ xamp_prev - V_next * (y - Z_prev) / (V_prev + channel.sigma2)
        U_next = 1 / (V_next + channel.sigma2)
        cov = 1 / (channel.Habs2_adj @ U_next)
        r = xamp_prev + cov * (channel.Hadj @ ((y - Z_next) * U_next))
        xamp_next, var_next = self.shrinkage(r, cov)
        return xamp_next, var_next, V_next, Z_next
    
    def bayesOOK(self, 
                  r: torch.Tensor, 
                  cov: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            r (torch.Tensor): _description_
            cov (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        eta = (self.P0 / self.Ps) * torch.exp((1 - 2*r) / cov) + self.tol
        exp = 1 / ( 1 + eta )
        var = exp - exp**2
        return exp, var
    
class BAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.device = config.device
        self.insize = config.insize
        self.sparsity = config.sparsity
        self.datatype = config.datatype
        self.N_Layers = config.N_Layers
        
        self.layers = nn.ModuleList([BAMPLayer(config) for _ in range(self.N_Layers)])
        self.loss = Loss(config)

    def forward(self,
                x:torch.Tensor, 
                y: torch.Tensor, 
                channel: Channel, 
                ) -> Loss:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            Loss: _description_
        """
        self.loss.init()
        channel.generate_filter()
        xamp = torch.zeros(self.insize, device=self.device, dtype=self.datatype)
        var = torch.ones(self.insize, device=self.device, dtype=torch.float32)  * self.sparsity
        V = torch.zeros_like(y)
        Z = y
        for i, layer in enumerate(self.layers):
            xamp, var, V, Z = layer(y, xamp, var, V, Z, channel)
            self.loss(xamp, x)
        return self.loss
        
