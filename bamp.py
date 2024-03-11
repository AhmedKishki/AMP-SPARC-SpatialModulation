from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from shrink import Shrink
from loss import Loss

import numpy as np

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
        self.Nt, self.Na, self.Lin, self.B = config.Nt, config.Na, config.Lin, config.B
        self.K = config.K
        self.M = self.Nt // self.Na
        self.L = self.Na * self.Lin
        self.LM = self.L * self.M
        self.Ps, self.P0 = torch.tensor(config.Ps), torch.tensor(config.P0)
        self.symbols = torch.tensor(config.symbols, device=config.device)
        
        if config.mode in ['segmented', 'sparc']:
            self.denoiser = self.segmented_denoiser
        else:
            self.denoiser = self.random_denoiser

    def forward(self,
                T: Tracker
                )-> None:
        """_summary_

        Args:
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
    
    def segmented_denoiser(self, s: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        s = s.view(self.B, self.L, self.M, 1)
        tau = tau.view(self.B, self.L, self.M, 1) / 2
        x = (torch.tile(s / tau, dims=(1, 1, 1, self.K)) * self.symbols.conj()).real
        eta = torch.exp(x - x.abs().max())
        eta2 = eta.sum(dim=-1).sum(dim=2, keepdim=True)
        xmmse = (self.symbols * eta).sum(dim=-1) / eta.sum(dim=-1).sum(dim=2, keepdim=True)
        xmmse2 = torch.tile(xmmse.view(self.B, self.L, self.M, 1), dims=(1, 1, 1, self.K))
        var0 = xmmse.abs()**2 * (1 - eta.sum(dim=-1) / eta2)
        vars = (torch.abs(xmmse2 - self.symbols.view(1, 1, 1, -1))**2 * eta).sum(dim=-1) / eta2
        var = var0 + vars
        return xmmse.view(self.B, self.LM, 1).to(torch.complex64), var.view(self.B, self.LM, 1).to(torch.float32)
    
    def random_denoiser(self, 
                        r: torch.Tensor, 
                        cov: torch.Tensor
                        ) -> torch.Tensor:
        """_summary_

        Args:
            u (torch.Tensor): _description_
            v (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        G = lambda s: torch.exp(- torch.abs(r - s)**2 / cov )
        G0, Gs = G(0), G(self.symbols)
        norm = self.regularize_zero(self.P0 * G0 + self.Ps * torch.sum(Gs, dim=-1).unsqueeze(-1))
        exp = self.Ps * torch.sum(self.symbols * Gs, dim=-1).unsqueeze(-1) / norm
        var = self.Ps * torch.sum(self.symbols.abs()**2 * Gs, dim=-1).unsqueeze(-1) / norm - torch.abs(exp)**2
        return exp.to(torch.complex64), var.to(torch.float32)
    
    def regularize_zero(self, a):
        a[a==0.] = 1e-9
        return a
    
    def regularize(self, a: torch.Tensor):
        max = np.log(torch.finfo(a.dtype).max)
        a[a>=max] = max - 1
        return a
    
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
            prev = T.var
            layer(T)
            next = T.var
            if torch.allclose(next, prev):
                break
        self.L(T.xmap, T.xmmse, x, symbols, indices, t+1)
        return self.L
    
        
