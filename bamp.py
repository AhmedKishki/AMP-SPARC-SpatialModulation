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
        self.xmmse = torch.zeros_like(x)
        self.var = torch.ones_like(x, dtype=torch.float32)
        self.v = torch.zeros_like(y)
        self.z = y
        self.xmap = None
        self.u = self.v + self.sigma2
        
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
        T.v = T.abs2 @ T.var
        T.z = T.H @ T.xmmse - T.v * (T.y - T.z) / T.u
        T.u = T.v + T.sigma2
        cov = 1 / (T.abs2T @ (1 / T.u))
        T.xmap = T.xmmse + cov * (T.adj @ ((T.y - T.z) / T.u))
        T.xmmse, T.var = self.denoiser(T.xmap, cov)
    
class BAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.E = config.Na / config.Nr
        
        self.layers = nn.ModuleList([BAMPLayer(config) for _ in range(config.N_Layers)])
        self.L = Loss(config)

    def forward(self,
                H: torch.Tensor,
                y: torch.Tensor,
                SNR: float,
                x: torch.Tensor, 
                symbols,
                indices
                ) -> Loss:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            xmmse: _description_
        """
        T = Tracker(x, y, H, self.E / SNR)
        self.L.dump()
        for t, layer in enumerate(self.layers):
            var_prev = T.var
            layer(T)
            var_next = T.var
            if torch.allclose(var_next, var_prev):
                break
        self.L(T.xmap, T.xmmse, x, symbols, indices, t+1)
        return self.L
    
        
