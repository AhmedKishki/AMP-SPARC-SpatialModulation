from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F
import math
from config import Config
from channel import Channel
from shrink import Shrink


class AMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.Lin = config.Lin
        self.Nt, self.Na, self.Nr = config.Nt, config.Na, config.Nr
        self.P0, self.Ps = config.P0, config.Ps
        self.tol = 1e-10
        
        self.shrinkage = Shrink(config, "lasso")
        
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
        print(channel.Habs2.dtype, var_prev.dtype)
        V_next = channel.Habs2 @ var_prev
        Z_next = channel.H @ xamp_prev - V_next * (y - Z_prev) / (V_prev + 1)
        U_next = 1 / (V_next + 1)
        cov = 1 / (channel.Habs2_adj @ U_next)
        r = xamp_prev + cov * (channel.Hadj @ ((y - Z_next) * U_next))
        xamp_next, var_next = self.shrinkage(r, cov)

        return xamp_next, var_next, V_next, Z_next
        
    
class AMP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.device = config.device
        self.insize = config.insize
        self.B, self.Lin, self.Nt, self.Na = config.B, config.Lin, config.Nt, config.Na
        self.sparsity = config.sparsity
        self.datatype = config.datatype
        self.N_Layers = config.N_Layers
        self.nz = config.nonzero_symbols
        self.xlen = config.xlen
        self.N, self.M = config.N, config.M
        
        self.layers = nn.ModuleList([AMPLayer(config) for _ in range(self.N_Layers)])
        self.MSE = nn.MSELoss()
        
    def forward(self, 
                x:torch.Tensor, 
                y: torch.Tensor, 
                channel: Channel, 
                ) -> Tuple[torch.Tensor, List]:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_
            sigma2_N (float): _description_

        Returns:
            Tuple[torch.Tensor, List]: _description_
        """
        xamp = torch.zeros(self.insize, device=self.device, dtype=self.datatype)
        var = torch.ones(self.insize, device=self.device, dtype=torch.float32) * self.M / self.N
        V = torch.zeros_like(y)
        Z = y
        mse_loss = [10*math.log10(self.sparsity)]
        error_loss = [self.nz]
        for i, layer in enumerate(self.layers):
            xamp, var, V, Z = layer(y, xamp, var, V, Z, channel)
            mse = 10*torch.log10(self.MSE(xamp, x))
            error = torch.count_nonzero(torch.round(xamp).int() - x.int())
            mse_loss.append(mse.item())
            error_loss.append(error.item())

        return xamp, mse_loss, error_loss