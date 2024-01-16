import torch
from torch import nn
import numpy as np

from config import Config
from loss import Loss

class Tracker:
    def __init__(self, 
                W: torch.Tensor, 
                A: torch.Tensor, 
                y:torch.Tensor, 
                sigma2:float, 
                x:torch.Tensor
                ):
        self.x = x
        self.y = y
        self.W = W
        self.A = A
        self.z = y
        self.psi = torch.ones(x.size(0), W.size(1), 1, dtype=torch.float32, device=x.device)
        self.phi = torch.ones(y.size(0), W.size(0), 1, dtype=torch.float32, device=y.device) * torch.inf
        self.xmmse = torch.zeros_like(x)
        self.sigma2 = sigma2
        self.xmap = None
        
class SCAMPLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.B = config.B
        self.Na = config.Na
        self.M = config.Nt // config.Na
        self.Mc = config.Nt
        self.Mr = config.Nr
        self.L = config.Na * config.Lin
        self.Lc = config.Lin
        self.Lr = config.Lout
        self.n = self.Mr * self.Lr
        self.LM = self.Mc * self.Lc
        self.K = config.K
        self.symbols = torch.tensor(config.symbols, device=config.device)
        
    def forward(self, T: Tracker) -> torch.Tensor:
        # Residual var - noise var (length Lr)
        gma = T.W @ T.psi / self.Lc # B, Lr, 1
        # Modified residual z
        b = gma / T.phi # B, Lr, 1
        T.z = T.y - T.A @ T.xmmse + b.repeat_interleave(self.Mr, dim=1) * T.z # B, n, 1
        # Residual variance phi
        T.phi = T.sigma2 + gma # B, Lr, 1
        # Effective noise variance tau
        tau = self.L / (T.W.T @ ( 1 / T.phi )) / self.Mr # B, Lc, 1
        tau_use = tau.repeat_interleave(self.Mc, dim=1) # B, LM, 1
        phi_use = T.phi.repeat_interleave(self.Mr, dim=1) # B, n, 1
        # Update message vector xmmse 
        T.xmap = T.xmmse + tau_use * (T.A.adjoint() @ (T.z / phi_use)) # B, LM, 1
        T.xmmse = self.denoiser(T.xmap, tau_use) # B, LM, 1
        # Update MMSE estimate
        T.psi = 1 - (T.xmmse.abs()**2).view(self.B, self.Lc, self.Mc).sum(dim=-1, keepdim=True) / self.Na # B, Lc, 1
        
    def denoiser(self, s: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        s = s.view(self.B, self.L, self.M, 1)
        tau = tau.view(self.B, self.L, self.M, 1) / 2
        x = (torch.tile(s / tau, dims=(1, 1, 1, self.K)) * self.symbols.conj()).real
        eta = torch.exp(x - x.abs().max())
        eta2 = self.symbols * eta
        xmmse = eta2.sum(dim=-1) / eta.sum(dim=-1).sum(dim=2, keepdim=True)
        return xmmse.view(self.B, self.LM, 1).to(torch.complex64)

class SCAMP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.E = config.Na / config.Nr
        self.layers = nn.ModuleList([SCAMPLayer(config) for _ in range(config.N_Layers)])
        self.L = Loss(config)
        
    def forward(self, 
                W: torch.Tensor, 
                A: torch.Tensor, 
                y:torch.Tensor, 
                SNR:float, 
                x:torch.Tensor, 
                symbol:np.ndarray, 
                index:np.ndarray):
        """_summary_

        Args:
            W (torch.Tensor): _description_
            A (torch.Tensor): _description_
            y (torch.Tensor): _description_
            SNR (float): _description_
            x (torch.Tensor): _description_
            symbol (np.ndarray): _description_
            index (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        T = Tracker(W, A, y, self.E / SNR, x)
        self.L.dump()
        for t, layer in enumerate(self.layers):
            psi_prev = T.psi
            layer(T)
            psi_next = T.psi
            if torch.allclose(psi_next, psi_prev):
                break
        self.L(T.xmap, T.xmmse, x, symbol, index, t+1)
        return self.L