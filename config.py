import torch
import numpy as np

class Config:
    def __init__(self,
                 batch : int,
                 N_transmit_antenna : int,
                 N_active_antenna : int,
                 N_receive_antenna : int,
                 block_length: int,
                 channel_length : int,
                 iterations: int,
                 alphabet: str = 'OOK',
                 channel_profile : str = 'exponential',
                 channel_truncation : bool = 'trunc',
                 device : str = 'cuda'
                ) -> None:
        """_summary_

        Args:
            Batch (int): _description_
            N_transmit_antenna (int): _description_
            N_active_antenna (int): _description_
            N_receive_antenna (int): _description_
            block_length (int): _description_
            channel_length (int): _description_
            VAMP_iterations (int): _description_
            full_svd (bool, optional): _description_. Defaults to False.
            channel_profile (str, optional): _description_. Defaults to 'exponential'.
            is_complex (bool, optional): _description_. Defaults to False.
            channel_truncation (bool, optional): _description_. Defaults to 'trunc'.
            device (str, optional): _description_. Defaults to 'cuda'.
        """
        
        assert channel_profile in ['exponential', 'uniform'], "channel_profile has to be 'exponential' or 'uniform'"
        assert channel_truncation in ['trunc', 'tail', 'cyclic'], "channel_truncation has to be 'trunc', 'tail' or 'cyclic'"
        assert channel_length > 0, "channel_length needs to be at least 1"
        
        self.device = device
        
        # Architecture
        self.B, self.Lin = batch, block_length
        self.Nt, self.Na, self.Nr = N_transmit_antenna, N_active_antenna, N_receive_antenna
        self.sparsity = self.Na / self.Nt
        
        # Channel
        self.Lh = channel_length
        self.profile = channel_profile
        self.trunc = channel_truncation
        self.Lout = self.Lin + self.Lh - 1
        if channel_truncation != 'tail':
            self.Lout = self.Lin
        self.ISI = False
        if self.Lh > 1:
            self.ISI = True
        if channel_profile == 'exponential':
            self.pdp = np.exp(-np.arange(channel_length))
        elif channel_profile == 'uniform':
            self.pdp = np.ones(channel_length)
        self.pdp = self.pdp / np.sum(self.pdp)
        self.M = self.Nt * self.Lin
        self.N = self.Nr * self.Lout
        self.Hsize = self.N, self.M
        
        # Data Generator
        self.insize = self.B, self.M, 1
        self.xlen = self.B * self.M
        self.outsize = self.B, self.N, 1
        self.ylen = self.B * self.N
        self.Ns = self.B * self.Lin * self.Na # number of symbols
        self.N0 = self.B * self.Lin * self.Nt # number of zeros
        self.is_complex = False
        self.Ps = self.sparsity
        self.P0 = 1 - self.Ps
        self.alphabet = alphabet
        
        if alphabet == 'OOK':
            self.symbols = [1]
            self.datatype = torch.float32
            self.npdatatype = np.float32
        
        elif alphabet == 'BPSK':
            self.symbols = [-1, 1]
            self.Ps = self.Ps / 2
            self.datatype = torch.float32
            self.npdatatype = np.float32
        
        elif alphabet == 'QPSK':
            self.symbols = [1+0j, -1+0j, 0+1j, 0-1j]
            self.Ps = self.Ps / 4
            self.is_complex = True
            self.datatype = torch.complex64
            self.npdatatype = np.complex64
        
        elif alphabet == "8PSK":
            self.symbols = [np.exp((2 * np.pi * 1j / 8) * n) for n in range(8)]
            self.Ps = self.Ps / 8
            self.is_complex = True
            self.datatype = torch.complex64
            self.npdatatype = np.complex64
        
        elif alphabet == "16PSK":
            self.symbols = [np.exp((2 * np.pi * 1j / 16) * n) for n in range(16)]
            self.Ps = self.Ps / 16
            self.is_complex = True
            self.datatype = torch.complex64
            self.npdatatype = np.complex64
        
        elif alphabet == "16QAM":
            self.symbols = [1+1j, -1+1j, -1+1j, -1-1j, 3+1j, -3+1j, 3-1j, -3-1j, 3+3j, -3+3j, 3-3j, -3-3j]
            self.Ps = self.Ps / 16
            self.is_complex = True
            self.datatype = torch.complex64
            self.npdatatype = np.complex64
        
        else:
            raise NameError('alphabet should be one of: OOK, BPSK, QPSK, 8PSK, 16PSK or 16QAM')
        
        self.N_symbols = len(self.symbols)
        self.signal_power = np.sqrt(np.sum(np.abs(self.symbols)**2) / self.N_symbols)
        self.symbols = np.array(self.symbols, dtype=self.npdatatype) / self.signal_power
        
        # AMP
        self.N_Layers = iterations
        self.epsilon = 1/self.sparsity - 1
        self.lmda = None
        
        # save
        self.name = f'M={self.alphabet},Nt={self.Nt},Na={self.Na},Nr={self.Nr},Lh={self.Lh},Lb={self.Lh}'
        