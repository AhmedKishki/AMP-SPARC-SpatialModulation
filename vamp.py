from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from shrink import Shrink
from loss import Loss

import numpy as np

class Tracker:
    def __init__(self, U:torch.Tensor, s:torch.Tensor, Vh:torch.Tensor, y:torch.Tensor, x:torch.Tensor, sigma2:float, sparsity: float):
        self.U = U
        self.Uh = U.adjoint()
        self.s = s
        self.s2 = s**2
        self.Vh = Vh
        self.V = Vh.adjoint()
        self.noise_var = sigma2
        self.y_tilde = self.Uh @ y / s.view(-1, 1)
        self.xmap = torch.zeros_like(x)
        self.var = torch.ones_like(x, dtype=torch.float32)
        self.r_tilde = torch.ones_like(x) * sparsity
        self.sigma2_tilde = (sparsity)**2 * (1-sparsity) + (1-sparsity)**2 * sparsity
        self.xmmse = None

class VAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.Nt, self.Nr = config.Nt, config.Nr
        self.denoiser = Shrink(config, 'bayes')
        
        self.var_ratio_min = torch.tensor(1.0e-5)
        self.var_ratio_max = 1.0 - self.var_ratio_min
        self.var_min = torch.tensor(1.0e-9)
        self.var_max = torch.tensor(1.0e5)
        
    def forward(self,
                T: Tracker):
        """_summary_

        Args:
            T (Tracker): _description_

        Returns:
            _type_: _description_
        """        
        var_ratio = T.noise_var / T.sigma2_tilde
        q = T.Vh @ T.r_tilde
        scale = 1 / (T.s2 + var_ratio)
        
        x_tilde = scale.view(-1, 1) * (T.y_tilde + var_ratio * q)
        varLMMSE = scale.mean() * T.noise_var
        x_tilde = T.V @ (x_tilde - q) + T.r_tilde
        x_tilde_var = self.Nr/self.Nt * varLMMSE + (1 - self.Nr/self.Nt) * T.sigma2_tilde

        alpha = x_tilde_var / T.sigma2_tilde
        alpha = torch.max(alpha, self.var_ratio_min)
        alpha = torch.min(alpha, self.var_ratio_max)
        
        T.xmap = (x_tilde - alpha * T.r_tilde) / (1 - alpha)
        sigma2 = alpha/(1-alpha)*T.sigma2_tilde
        sigma2 = torch.max(sigma2, self.var_min)
        sigma2 = torch.min(sigma2, self.var_max)
        
        T.xmmse, T.var = self.denoiser(T.xmap, sigma2)
        dxdr = T.var.mean() / sigma2
        dxdr = torch.max(dxdr, self.var_ratio_min)
        dxdr = torch.min(dxdr, self.var_ratio_max)
        
        T.r_tilde = (T.xmmse - dxdr * T.xmap) / (1 - dxdr)
        T.sigma2_tilde = sigma2 * dxdr / (1 - dxdr)
        T.sigma2_tilde = torch.max(T.sigma2_tilde, self.var_min)
        T.sigma2_tilde = torch.max(T.sigma2_tilde, self.var_min)
    
class VAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.E = config.Na
        self.sparsity = config.Nt / config.Na
        self.layers = nn.ModuleList([VAMPLayer(config) for _ in range(config.N_Layers)])
        self.L = Loss(config)

    def forward(self,
                U: torch.Tensor,
                s: torch.Tensor,
                Vh: torch.Tensor,
                y: torch.Tensor,
                SNR: float,
                x: torch.Tensor, 
                symbols: np.ndarray,
                indices: np.ndarray
                ) -> Loss:
        """_summary_

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_
            channel (Channel): _description_

        Returns:
            xmmse: _description_
        """
        T = Tracker(U, s, Vh, y, x, self.E / SNR, self.sparsity)
        self.L.dump()
        for t, layer in enumerate(self.layers):
            layer(T)
        self.L(T.xmap, T.xmmse, x, symbols, indices, t+1)
        return self.L