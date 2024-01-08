from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from shrink import Shrink
from loss import Loss

class Tracker:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, H: torch.Tensor, sigma2: float):
        self.y = y
        self.H = H
        self.sigma2 = sigma2
        self.adj = H.adjoint()
        self.abs2 = H.abs()**2
        self.abs2T = self.abs2.T
        self.xamp = torch.zeros_like(x)
        self.var = torch.ones_like(x, dtype=torch.float32)
        self.V = torch.zeros_like(y)
        self.Z = y
        
class BAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.denoiser = Shrink(config, "bayes")
        
    def forward(self,
                T: Tracker
                )-> None:
        """_summary_

        Args:
            y (torch.Tensor): _description_
            T (Tracker): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        U = 1 / (T.V + T.sigma2)
        T.V = T.abs2 @ T.var
        T.Z = T.H @ T.xamp - T.V * (T.y - T.Z) * U
        U = 1 / (T.V + T.sigma2)
        cov = 1 / (T.abs2T @ U)
        r = T.xamp + cov * (T.adj @ ((T.y - T.Z) * U))
        T.xamp, T.var = self.denoiser(r, cov)
        return T.xamp
    
class BAMP(nn.Module):
    def __init__(self, config: Config, save_all_layers: bool = True) -> None:
        super().__init__()
        self.E = config.Na / config.Nr
        self.save = save_all_layers
        self.device = config.device
        self.sparsity = config.sparsity
        self.N_Layers = config.N_Layers
        
        self.layers = nn.ModuleList([BAMPLayer(config) for _ in range(self.N_Layers)])
        self.L = Loss(config)

    def forward(self,
                x:torch.Tensor, 
                y: torch.Tensor,
                H: torch.Tensor,
                SNR: float,
                symbols: torch.Tensor,
                indices: torch.Tensor
                ) -> Loss:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            xamp: _description_
        """
        sigma2 = self.E / SNR 
        T = Tracker(x, y, H, sigma2)
        self.L.dump()
        for i, layer in enumerate(self.layers):
            xamp = layer(T)
            self.L(xamp, x, symbols, indices)
        return self.L
    
        
