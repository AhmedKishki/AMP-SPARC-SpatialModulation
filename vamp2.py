from typing import Tuple, List
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from shrink import Shrink
from loss import Loss

import numpy as np

class Tracker:
    def __init__(self, U:torch.Tensor, s:torch.Tensor, Vh:torch.Tensor, y:torch.Tensor, x:torch.Tensor, sigma2:float):
        self.U = U
        self.Uh = U.adjoint()
        self.s = s.view(-1, 1)
        self.s2 = self.s**2
        self.Vh = Vh
        self.V = Vh.adjoint()
        self.sigma2 = sigma2
        self.gamma = torch.tensor(1.0)
        self.y_tilde = (self.Uh @ y) / self.s
        self.r = torch.zeros_like(x)
        self.var = torch.ones_like(x, dtype=torch.float32)
        self.xmmse = torch.zeros_like(x)
        self.eta = Vh.size(1) / s.size(0)

class VAMPLayer(nn.Module):
    def __init__(self, config: Config, damping=0.97) -> None:
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
        self.symbols = torch.tensor(config.symbols, device=config.device)
        
        if config.mode in ['segmented', 'sparc']:
            self.denoiser = self.segmented_denoiser
        else:
            self.denoiser = Shrink(config, 'bayes')
        
        self.var_min = torch.tensor(1.0e-11)
        self.var_max = torch.tensor(1.0e11)
        self.rho = damping
        
    def forward(self,
                T: Tracker):
        """Direct implementation of Rangan (with damping)

        Args:
            T (Tracker): _description_

        Returns:
            _type_: _description_
        """
        xmmse, T.var = self.denoiser(T.r, T.gamma)
        T.xmmse = self.rho * xmmse + (1 - self.rho) * T.xmmse
        alpha = T.var.mean() * T.gamma
        
        r_tilde = (T.xmmse - alpha * T.r) / (1 - alpha)
        gamma_tilde = T.gamma * (1 - alpha) / alpha
        gamma_tilde = torch.max(gamma_tilde, self.var_min)
        gamma_tilde = torch.min(gamma_tilde, self.var_max)
        
        d = T.s2 / (T.s2 + T.sigma2 * gamma_tilde) 
        gamma = gamma_tilde * d.mean() / (T.eta - d.mean())
        T.gamma = self.rho * gamma + (1 - self.rho) * T.gamma
        gamma = torch.max(gamma_tilde, self.var_min)
        gamma = torch.min(gamma_tilde, self.var_max)
        
        T.r = r_tilde + T.eta * T.V @ ((d / d.mean()) * (T.y_tilde - T.Vh @ r_tilde))
        
    def segmented_denoiser(self, s: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        s = s.view(self.B, self.L, self.M, 1)
        # tau = tau.view(self.B, self.L, self.M, 1) / 2
        x = (torch.tile(s / tau, dims=(1, 1, 1, self.K)) * self.symbols.conj()).real
        eta = torch.exp(x - x.abs().max())
        eta2 = self.symbols * eta
        eta3 = self.symbols.abs()**2 * eta
        xmmse = eta2.sum(dim=-1) / eta.sum(dim=-1).sum(dim=2, keepdim=True)
        var = eta3.sum(dim=-1) / eta.sum(dim=-1).sum(dim=2, keepdim=True) - xmmse.abs()**2
        return xmmse.view(self.B, self.LM, 1).to(torch.complex64), var.view(self.B, self.LM, 1).to(torch.float32)
    
    def regularize(self, a: torch.Tensor):
        max = np.log(torch.finfo(a.dtype).max)
        a[a>=max] = max - 1
        return a
        
class VAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.E = config.Na / config.Nr
        self.sparsity = config.Na / config.Nt
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
        T = Tracker(U, s, Vh, y, x, self.E / SNR)
        self.L.dump()
        for t, layer in enumerate(self.layers):
            prev = T.var
            layer(T)
            next = T.var
            if torch.allclose(next, prev):
                break
        self.L(T.r, T.xmmse, x, symbols, indices, t+1)
        # np.savetxt('r.txt', T.r.view(-1).cpu().numpy())
        # np.savetxt('xmmse.txt', T.xmmse.view(-1).cpu().numpy())
        # np.savetxt('var.txt', T.var.view(-1).cpu().numpy())
        return self.L