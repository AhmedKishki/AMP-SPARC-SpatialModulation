from typing import Tuple
import torch
from torch.nn import functional as F
import numpy as np

from config import Config

class Channel:
    def __init__(self, 
                 config: Config
                 ) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.device = config.device
        self.B, self.Lin = config.B, config.Lin
        self.Nt, self.Na, self.Nr = config.Nt, config.Na, config.Nr
        self.Lh = config.Lh
        self.trunc = config.trunc
        self.npdtype = config.npdatatype
        self.datatype = config.datatype
        self.pdp = config.pdp
        self.Lout = config.Lout
        self.is_complex = config.is_complex
        self.N, self.M = config.N, config.M
        
        if self.is_complex:
            self.awgn = self.complex_awgn
        else:
            self.awgn = self.real_awgn
        
        if config.ISI:
            self.generate_channel = self.generate_ISI_channel
        else:
            self.generate_channel = self.generate_constant_channel
            
    def __call__(self, x: torch.Tensor, SNR: float) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): size: Batch, N_transmit_antenna * Lin, 1.
            SNR (float): _description_

        Returns:
            torch.Tensor: size depends on channel_truncation.
            if channel_truncation is 'trunc':
                size: Batch, N_receive_antenna * Lin, 1.
            
            if 'tail':
                size: Batch, N_receive_antenna * (Lin + Lh - 1), 1
            
            if 'cyclic':
                size: Batch, N_receive_antenna * Lin, 1.
        """
        self.SNR = SNR
        self.sigma2 = self.Na / self.SNR
        self.sigma = np.sqrt(self.sigma2 / self.Nr / 2)
        return self.H @ x + self.awgn()
    
    def capacity(self):
        return 0.5 * np.log2(1 + self.SNR)
    
    def complex_awgn(self) -> torch.Tensor:
        """
        generate awgn for MIMO

        Args:
            SNR (_type_): _description_

        Returns:
            torch.Tensor: _description_
        """
        awgn_r = torch.normal(mean=0., std=self.sigma, size=(self.B, self.Nr * self.Lout, 1), device=self.device)
        awgn_i = torch.normal(mean=0., std=self.sigma, size=(self.B, self.Nr * self.Lout, 1), device=self.device)
        return awgn_r + 1j * awgn_i
    
    def real_awgn(self) -> torch.Tensor:
        """
        generate awgn for MIMO

        Args:
            SNR (_type_): _description_

        Returns:
            torch.Tensor: _description_
        """
        return torch.normal(mean=0., std=self.sigma, size=(self.B, self.Nr * self.Lout, 1), device=self.device)
    
    def generate_constant_channel(self) -> None:
        """_summary_
        """
        if self.is_complex:
            Hr = torch.ones(self.Nt, device=self.device)
            Hi = torch.ones(self.Nt, device=self.device)
            H = Hr + 1j * Hi
        else:
            H = torch.ones(self.Nt)
        
        self.H = H.repeat(self.Nr*self.Lin, self.Lin)
            
    def generate_random_channel(self)->None:
        pass
        
    def generate_ISI_channel(self) -> None:
        """
        generate MIMO Toeplitz matrix for frequency-selective channel

        if channel_truncation is 'trunc':
            size: N_receive_antenna, N_transmit_antenna
        
        if 'tail':
            size: N_receive_antenna, N_transmit_antenna x (Lh - 1), 1
        
        if 'cyclic':
            size: N_receive_antenna, N_transmit_antenna
        """
        if self.is_complex:
            hr = np.random.normal(size=(self.Nr, self.Nt, self.Lh), scale = 1/np.sqrt(2*self.Nr))
            hi = np.random.normal(size=(self.Nr, self.Nt, self.Lh), scale = 1/np.sqrt(2*self.Nr))
            h = hr + 1j * hi
        else:
            h = np.random.normal(size=(self.Nr, self.Nt, self.Lh), scale = 1/np.sqrt(self.Nr))
        
        self.h = h * np.sqrt(self.pdp)
        
        H = np.zeros((self.Lin*self.Nr, self.Lin*self.Nt), dtype=self.npdtype)
        for l in np.arange(self.Lh):
            H += np.kron(np.eye(self.Lin, self.Lin, -l), h[:,:, l])
        
        if self.trunc == 'tail':
            # add post-transient samples making frame longer (more rows in H)
            H_append = np.zeros((self.Nr * (self.Lh - 1), self.Nt * self.Lin), dtype=self.npdtype)
            tail = H[-self.Nr:, -self.Nt*self.Lh:-self.Nt]
            for l in np.arange(self.Lh-1):
                H_append[l*self.Nr:(l+1)*self.Nr, -self.Nt*(self.Lh-l-1):] = tail[:, :self.Nt*(self.Lh-l-1)]
                H = np.block([[H],[H_append]])
        
        elif self.trunc == 'cyclic':
            # cyclic convolution --> give H cyclic Toeplitz structure (fill upper right corner)
            tail = H[-self.Nr:, -self.Nt*self.Lh:-self.Nt]
            for Lh in np.arange(self.Lh-1):
                H[Lh*self.Nr:(Lh+1)*self.Nr, -self.Nt*(self.Lh-Lh-1):] = tail[:, :self.Nt*(self.Lh-Lh-1)]
        self.H = torch.tensor(H, dtype=self.datatype, requires_grad=False, device=self.device)
    
    def generate_svd(self) -> None:
        """_summary_
        """
        U, s, Vh = torch.linalg.svd(self.H.cpu(), full_matrices=False)
        self.U, self.s, self.Vh = U.to(self.device), s.to(self.device).to(self.datatype), Vh.to(self.device)
        self.Uh = self.U.adjoint()
        self.s2 = self.s ** 2
        self.V = self.Vh.adjoint()
        self.R = self.s.numel()
        
    def generate_filter(self) -> None:
        """_summary_
        """
        self.Hadj = self.H.adjoint()
        self.Habs2 = torch.abs(self.H)**2
        self.Habs2_adj = self.Habs2.adjoint()
            
if __name__ == "__main__":
    config = Config(1, 100, 5, 50, 4, 3)
    channel = Channel(config)
    channel.generate_channel()