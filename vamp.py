from typing import Tuple
import torch
from torch import nn
import numpy as np

from config import Config
from shrink import Shrink
from loss import Loss

class Tracker:
    def __init__(self, x, y, H, sigma2, sparsity):
        self.noise_var = sigma2
        self.U, self.s, self.Vh = torch.linalg.svd(H, full_matrices=False)
        self.s2 = self.s**2
        self.ytilde = self.U @ y / self.s
        self.rtilde = torch.ones_like(self.ytilde) * sparsity
        self.sigma2tilde = (0-sparsity)**2 * (1-sparsity) + (1-sparsity)**2 * sparsity
        
        
class VAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """
        VAMP iteration (layer)

        Args:
            config (Config): _description_
        """
        super().__init__()
        
        
    def forward(self, 
                T: Tracker
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
        var_ratio = T.noise_var / T.sigma2tilde
        scale = T.s2 / ( T.s2 + T.noise_var )
        normLMMSE = 1 / scale.mean()
        Q = T.Vh @ T.rtilde
        xtilde = scale * (T.ytilde - Q)

        return 

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
        self.E = config.Na / config.Nr
        self.sparsity = config.sparsity
        self.N_Layers = config.N_Layers
        
        # setup
        self.layers = nn.ModuleList([VAMPLayer(config) for _ in range(self.N_Layers+1)])
        self.L = Loss(config)
        
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor, 
                H: torch.Tensor,
                SNR: float,
                symbols: np.ndarray,
                indices: np.ndarray,
                ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            torch.Tensor: _description_
        """
        T = Tracker(x, y, H, self.E / SNR, self.sparsity)
        self.L.dump()
        for i, layer in enumerate(self.layers):
            xamp = layer(T)
            self.L(xamp, x, symbols, indices)
        return self.L
         
if __name__ == "__main__":
    pass