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
        self.N_symbols = config.N_symbols
        self.Lh = config.Lh
        self.trunc = config.trunc
        self.npdtype = config.npdatatype
        self.datatype = config.datatype
        self.pdp = config.pdp
        self.Lout = config.Lout
        self.is_complex = config.is_complex
        
        if self.is_complex:
            self.awgn = self.complex_awgn
        else:
            self.awgn = self.real_awgn
        
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
        return self.H @ x + self.awgn(SNR)
    
    def complex_awgn(self, SNR: float) -> torch.Tensor:
        """
        generate awgn for MIMO

        Args:
            SNR (_type_): _description_

        Returns:
            torch.Tensor: _description_
        """
        self.sigma2 = self.Na / self.N_symbols / SNR
        sigma = np.sqrt(self.Na / self.Nr / self.N_symbols / SNR / 2)
        awgn_r = torch.normal(mean=0., std=sigma, size=(self.B, self.Nr * self.Lout, 1), device=self.device)
        awgn_i = torch.normal(mean=0., std=sigma, size=(self.B, self.Nr * self.Lout, 1), device=self.device)
        return awgn_r + 1j * awgn_i
    
    def real_awgn(self, SNR: float) -> torch.Tensor:
        """
        generate awgn for MIMO

        Args:
            SNR (_type_): _description_

        Returns:
            torch.Tensor: _description_
        """
        self.sigma2 = self.Na / self.N_symbols / SNR
        sigma = np.sqrt(self.Na / self.Nr / self.N_symbols / SNR / 2)
        return torch.normal(mean=0., std=sigma, size=(self.B, self.Nr * self.Lout, 1), device=self.device)
    
    def generate(self, svd: bool = False) -> None:
        """_summary_

        Args:
            svd (bool, optional): _description_. Defaults to False.
        """
        self.H = self.generate_channel()
        self.Hadj = self.H.adjoint()
        self.Habs2 = torch.abs(self.H)**2
        self.Habs2_adj = self.Habs2.adjoint()
        
        if svd:
            self.U, self.s, self.Vh = self.generate_svd()
            self.Uh = self.U.adjoint()
            self.s2 = self.s ** 2
            self.V = self.Vh.adjoint()
        
    def generate_channel(self) -> torch.Tensor:
        """
        generate MIMO Toeplitz matrix for frequency-selective channel

        Returns:
            torch.Tensor: size depends on channel truncation:
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
        
        H = np.zeros((self.Lin*self.Nr, self.Lin*self.Nt), dtype=self.npdtype)
        for l in np.arange(self.Lh):
            H += np.kron(np.eye(self.Lin, self.Lin, -l), h[:,:, l]*np.sqrt(self.pdp[l]))
        
        if self.trunc == 'tail':
            # add post-transient samples making frame longer (more rows in H)
            H_append = np.zeros((self.Nr * (self.Lh - 1), self.Nt * self.Lin), dtype=H.dtype)
            tail = H[-self.Nr:, -self.Nt*self.Lh:-self.Nt]
            for l in np.arange(self.Lh-1):
                H_append[l*self.Nr:(l+1)*self.Nr, -self.Nt*(self.Lh-l-1):] = tail[:, :self.Nt*(self.Lh-l-1)]
                H = np.block([[H],[H_append]])
        
        elif self.trunc == 'cyclic':
            # cyclic convolution --> give H cyclic Toeplitz structure (fill upper right corner)
            tail = H[-self.Nr:, -self.Nt*self.Lh:-self.Nt]
            for Lh in np.arange(self.Lh-1):
                H[Lh*self.Nr:(Lh+1)*self.Nr, -self.Nt*(self.Lh-Lh-1):] = tail[:, :self.Nt*(self.Lh-Lh-1)]

        # # full convolution matrix (channel_truncation = 'tail')
        # Lout = self.Lh + self.Lin - 1
        # H = np.zeros((self.Nr, Lout, self.Nt, self.Lin), dtype=self.npdtype)
        # for nr in range(self.Nr):
        #     for nt in range(self.Nt):
        #         for l in range(self.Lh):
        #             H[nr, :, nt, :] += np.eye(Lout, self.Lin, -l, dtype=self.npdtype) * h[nr, nt, l]
        
        # if self.trunc == 'trunc':
        #     # delete transient rows
        #     Lout = self.Lin
        #     H = H[:, :Lout, :, :]
            
        # elif self.trunc == 'cyclic':
        #     # cyclic convolution
        #     Lout = self.Lin
        #     H_ = np.zeros((self.Nr, Lout, self.Nt, self.Lin))
        #     for nr in range(self.Nr):
        #         for nt in range(self.Nt):
        #             for l in range(self.Lh):
        #                 H_[nr, :, nt, :] += np.eye(Lout, self.Lin, self.Lin - self.Lh + l + 1) * h[nr, nt, - l - 1]
        #     H = H[:, :Lout, :, :] + H_
        # H = torch.tensor(np.reshape(H, (self.Nr * Lout, self.Nt * self.Lin)), dtype=self.datatype, requires_grad=False, device=self.device)
        return H
    
    def generate_svd(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """_summary_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        U, s, Vh = torch.linalg.svd(self.H.cpu(), full_matrices=self.full)
        return U.to(self.device), s.to(self.device), Vh.to(self.device)
            
if __name__ == "__main__":
    config = Config(1, 100, 5, 50, 4, 3)
    channel = Channel(config)
    print(channel.Lh)
    print(1/channel.S2)