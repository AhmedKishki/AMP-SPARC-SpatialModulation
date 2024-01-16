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
                 iterations: int = 20,
                 alphabet: str = 'OOK',
                 channel_profile : str = 'exponential',
                 channel_truncation : bool = 'trunc',
                 is_complex: bool = True,
                 device : str = 'cuda'
                ) -> None:
        """_summary_

        Args:
            N_transmit_antenna (int): _description_
            N_active_antenna (int): _description_
            N_receive_antenna (int): _description_
            block_length (int): _description_
            channel_length (int): _description_
            batch (int, optional): _description_. Defaults to 100.
            generator_mode (str, optional): _description_. Defaults to 'random'.
            iterations (int, optional): _description_. Defaults to 20.
            alphabet (str, optional): _description_. Defaults to 'OOK'.
            channel_profile (str, optional): _description_. Defaults to 'exponential'.
            channel_truncation (bool, optional): _description_. Defaults to 'trunc'.
            is_complex (bool, optional): _description_. Defaults to True.
            device (str, optional): _description_. Defaults to 'cuda'.

        Raises:
            ValueError: _description_
        """
        assert channel_profile in ['exponential', 'uniform', 'random'], "channel_profile has to be 'exponential' or 'uniform'"
        assert channel_truncation in ['trunc', 'tail', 'cyclic'], "channel_truncation has to be 'trunc', 'tail' or 'cyclic'"
        assert channel_length > 0, "channel_length needs to be at least 1"
        assert generator_mode in ['segmented', 'random', 'sparc'], "generator_mode needs to be 'segmented' or 'random' or 'sparc'"
        assert alphabet in ['OOK','BPSK','4ASK','QPSK','8PSK','16PSK','16QAM'], "alphabet has to be 'OOK','BPSK','4ASK','QPSK','8PSK','16PSK' or'16QAM'"
        
        self.device = device
        
        # Architecture
        self.B, self.Lin = batch, block_length
        self.Nt, self.Na, self.Nr = N_transmit_antenna, N_active_antenna, N_receive_antenna
        self.sparsity = self.Na / self.Nt
        self.mode = generator_mode
        
        # Channel
        self.is_complex = is_complex
        self.Lh = channel_length
        self.profile = channel_profile
        self.trunc = channel_truncation
    
        if channel_truncation != 'tail':
            self.Lout = self.Lin
        else:
            self.Lout = self.Lin + self.Lh - 1
        
        if self.Lh > 1:
            self.ISI = True
        else:
            self.ISI = False
        
        # Data Generator
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
            
        elif alphabet == '4ASK':
            self.symbols = [-3, -1, 1, 3]
            self.gray = [0, 1, 3, 2]
            self.Ps = self.Ps / 4

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
            self.symbols = [1+1j, 1-1j, -1+1j, -1-1j, 3+1j, 3-1j, -3+1j, -3-1j, 3+3j, 3-3j, -3+3j, -3-3j, 1+3j, -1+3j, -1+3j, -1-3j]
            self.gray = [0, 1, 13, 7, 8, 9, 2, 15, 12, 11, 5, 10, 14, 3, 6, 4]
            self.Ps = self.Ps / 16
            self.is_complex = True
        
        self.symbols = np.array(self.symbols) / np.sqrt(np.mean(np.abs(self.symbols)**2))
        self.K = len(self.symbols)
        self.symbol_bits = int(np.log2(self.K))
        
        if self.mode == 'random':
            self.index_bits = np.log2(np.prod([1 + (self.Nt - self.Na)/j for j in range(1, self.Na+1)]))
            self.info_bits = self.symbol_bits + self.index_bits
            self.code_rate = self.Lin * self.info_bits / self.Nr / self.Lout
        
        elif self.mode == 'segmented':
            assert self.Nt % self.Na == 0,'Na must divide Nt'
            self.index_bits = self.Na * np.log2(self.Nt / self.Na)
            self.info_bits = self.symbol_bits + self.index_bits
            self.code_rate = self.Lin * self.info_bits / self.Nr / self.Lout
            
        elif self.mode == 'sparc':
            assert self.Nt % self.Na == 0,'Na must divide Nt'
            self.M = self.Nt // self.Na
            self.Mc = self.Nt
            self.Mr = self.Nr
            self.L = self.Na * self.Lin
            self.Lc = self.Lin
            self.Lr = self.Lout
            self.n = self.Nr * self.Lout
            self.index_bits = self.Na * np.log2(self.M)
            self.symbol_bits = int(np.log2(self.K))
            self.inner_code_rate = self.Na * np.log2(self.M * self.K) / self.Mr
            self.code_rate = self.Lc * self.inner_code_rate / self.Lr
        
        # AMP
        self.N_Layers = iterations
        self.kappa = self.Lout / self.Lin
        self.min_amp_snr = 1 / (self.kappa * (1 / (np.exp(2*self.code_rate) - 1) - 1 /self.Lh))
        
        # simulation
        self.min_snr = 2**self.code_rate - 1
        self.min_snr_dB = 10*np.log10(self.min_snr)
        self.shannon_limit_dB = self.min_snr_dB - 10*np.log10(self.code_rate)
        
        # save
        self.name = f'{self.alphabet},{self.mode}/{self.profile},{self.trunc}/Nt={self.Nt},Na={self.Na},Nr={self.Nr},Lh={self.Lh},Lin={self.Lin}'