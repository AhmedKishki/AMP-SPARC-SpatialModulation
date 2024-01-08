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
        self.Lout = config.Lout
        self.sparsity = config.sparsity
        self.is_complex = config.is_complex
        
        if config.profile == 'exponential':
            self.pdp = np.exp(-np.arange(self.Lh))
        elif config.profile == 'uniform':
            self.pdp = np.ones(self.Lh)
        self.pdp = self.pdp / np.sum(self.pdp)
        
        if self.is_complex:
            self.dtype = torch.complex64
            self.npdtype = np.complex64
        else:
            self.dtype = torch.float32
            self.npdtype = np.float32
    
    def capacity(self, SNR) -> float:
        """AWGN capacity

        Returns:
            _type_: _description_
        """
        return np.log2(1 + SNR)
        
    def generate_channel(self) -> torch.Tensor:
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
            hr = np.random.normal(size=(self.Nr, self.Nt, self.Lh))
            hi = np.random.normal(size=(self.Nr, self.Nt, self.Lh))
            h = hr + 1j * hi
        else:
            h = np.random.normal(size=(self.Nr, self.Nt, self.Lh))
            
        h = h * np.sqrt(self.pdp) / np.sqrt(2 * self.Nr) 
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
        return torch.tensor(H, dtype=self.dtype, requires_grad=False, device=self.device)
       
    def awgn(self, SNR) -> torch.Tensor:
        """
        generate awgn for MIMO

        Args:
            SNR (_type_): _description_

        Returns:
            torch.Tensor: _description_
        """
        awgn_r = torch.normal(mean=0., std=1., size=(self.B, self.Nr * self.Lout, 1), device=self.device)
        awgn_i = torch.normal(mean=0., std=1., size=(self.B, self.Nr * self.Lout, 1), device=self.device)
        awgn = (awgn_r + 1j * awgn_i) * np.sqrt(self.Na / self.Nr / SNR / 2)
        return awgn
            
if __name__ == "__main__":
    config = Config(1, 100, 5, 50, 4, 3)
    channel = Channel(config)
    channel.generate_channel()