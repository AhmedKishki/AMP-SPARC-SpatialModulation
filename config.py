import torch
import numpy as np

class Config:
    def __init__(self,
                 N_transmit_antenna : int,
                 N_active_antenna : int,
                 N_receive_antenna : int,
                 block_length: int,
                 channel_length : int,
                 batch : int = 100,
                 generator_mode: str = 'random',
                 iterations: int =20,
                 alphabet: str = 'OOK',
                 channel_profile : str = 'exponential',
                 channel_truncation : bool = 'trunc',
                 is_complex: bool = True,
                 device : str = 'cuda'
                ) -> None:
        """_summary_

        Args:
            batch (int): _description_
            N_transmit_antenna (int): _description_
            N_active_antenna (int): _description_
            N_receive_antenna (int): _description_
            block_length (int): _description_
            channel_length (int): _description_
            iterations (int): _description_
            alphabet (str, optional): _description_. Defaults to 'OOK'.
            channel_profile (str, optional): _description_. Defaults to 'exponential'.
            channel_truncation (bool, optional): _description_. Defaults to 'trunc'.
            is_complex (bool, optional): _description_. Defaults to True.
            device (str, optional): _description_. Defaults to 'cuda'.

        Raises:
            NameError: _description_
        """
        assert channel_profile in ['exponential', 'uniform'], "channel_profile has to be 'exponential' or 'uniform'"
        assert channel_truncation in ['trunc', 'tail', 'cyclic'], "channel_truncation has to be 'trunc', 'tail' or 'cyclic'"
        assert channel_length > 0, "channel_length needs to be at least 1"
        assert generator_mode in ['sparc', 'random'], "generator_mode needs to be 'sparc' or 'random'"
        assert alphabet in ['OOK','BPSK','QPSK','8PSK','16PSK','16QAM'], "alphabet has to be 'OOK','BPSK','QPSK','8PSK','16PSK' or'16QAM'"
        
        self.device = device
        
        # Architecture
        self.B, self.Lin = batch, block_length
        self.Nt, self.Na, self.Nr = N_transmit_antenna, N_active_antenna, N_receive_antenna
        self.sparsity = self.Na / self.Nt
        self.mode = generator_mode
        
        # Channel
        self.is_complex = is_complex
        
        if is_complex:
            self.datatype = torch.complex64
            self.npdatatype = np.complex64
        else:
            self.datatype = torch.float32
            self.npdatatype = np.float32
            
        self.Lh = channel_length
        self.profile = channel_profile
        self.trunc = channel_truncation
        
        self.Lout = self.Lin + self.Lh - 1
        if channel_truncation != 'tail':
            self.Lout = self.Lin
        
        if self.Lh > 1:
            self.ISI = True
        else:
            self.ISI = False
        
        if channel_profile == 'exponential':
            self.pdp = np.exp(-np.arange(channel_length), dtype=self.npdatatype)
        elif channel_profile == 'uniform':
            self.pdp = np.ones(channel_length, dtype=self.npdatatype)
        self.pdp = self.pdp / np.sum(self.pdp)
        
        # Data Generator
        self.M = self.Nt * self.Lin
        self.N = self.Nr * self.Lout
        self.Hsize = self.N, self.M
        self.insize = self.B, self.M, 1
        self.xlen = self.B * self.M
        self.outsize = self.B, self.N, 1
        self.ylen = self.B * self.N
        self.Ns = self.B * self.Lin * self.Na # number of symbols
        self.N0 = self.B * self.Lin * (self.Nt - self.Na) # number of zeros
        self.Ps = self.sparsity
        self.P0 = 1 - self.Ps
        self.alphabet = alphabet
        self.modulated = True
        
        if alphabet == 'OOK':
            self.symbols = [1]
            self.gray = [1]
            self.modulated = False
        
        elif alphabet == 'BPSK':
            self.symbols = [-1, 1]
            self.gray = [0, 1]
            self.Ps = self.Ps / 2

        elif alphabet == 'QPSK':
            self.symbols = [1+0j, 0+1j, -1+0j, 0-1j]
            self.gray = [0, 1, 3, 2]
            self.Ps = self.Ps / 4
            self.is_complex = True
        
        elif alphabet == "8PSK":
            self.symbols = [np.exp((2 * np.pi * 1j / 8) * n) for n in range(8)]
            self.gray = [0, 1, 3, 2, 6, 7, 5, 4]
            self.Ps = self.Ps / 8
            self.is_complex = True
        
        elif alphabet == "16PSK":
            self.symbols = [np.exp((2 * np.pi * 1j / 16) * n) for n in range(16)]
            self.gray = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
            self.Ps = self.Ps / 16
            self.is_complex = True
        
        elif alphabet == "16QAM":
            self.symbols = [1+1j, -1+1j, -1+1j, -1-1j, 3+1j, -3+1j, 3-1j, -3-1j, 3+3j, -3+3j, 3-3j, -3-3j]
            self.gray = ...
            self.Ps = self.Ps / 16
            self.is_complex = True
        
        self.symbols = np.array(self.symbols, dtype=self.npdatatype) / np.sqrt(np.mean(np.abs(self.symbols)**2))
        self.modsize = len(self.symbols)
        
        if generator_mode == 'random':
            logkNtcNa = np.log2(self.modsize * np.prod([1 + (self.Nt - self.Na)/j for j in range(1, self.Na+1)]))
            self.code_rate = self.Lin * logkNtcNa / self.Nr / self.Lout
        
        elif generator_mode == 'sparc':
            if self.Nt % self.Na != 0:
                raise ValueError('Na must divide Nt')
            self.code_rate = self.Lin * self.Na * np.log2(self.modsize * self.Nt / self.Na) / self.Nr / self.Lout
        
        # AMP
        self.N_Layers = iterations
        self.epsilon = 1/self.sparsity - 1
        self.lmda = None
        
        # simulation
        self.min_snr = 4**self.code_rate - 1
        self.min_snr_dB = 10*np.log10(4**self.code_rate - 1)
        
        # save
        self.name = f'{self.alphabet}/Nt={self.Nt},Na={self.Na},Nr={self.Nr},Lh={self.Lh},Lb={self.Lin}'
        if self.trunc != 'trunc' or self.profile != 'exponential':
            self.name = f'{self.alphabet}/{self.trunc},{self.profile}/Nt={self.Nt},Na={self.Na},Nr={self.Nr},Lh={self.Lh},Lb={self.Lin}'