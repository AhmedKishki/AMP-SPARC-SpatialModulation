from typing import Tuple
import torch
import numpy as np
import json

from config import Config

class Loss:
    def __init__(self, config: Config) -> None:
        """_summary_

        Args:
            config (Config): _description_
        """
        super().__init__()
        self.B, self.Nt, self.Na, self.Nr, self.Lin = config.B, config.Nt, config.Na, config.Nr, config.Lin
        self.Ns, self.sparsity = config.Ns, config.sparsity
        self.gray = config.gray
        self.symbols = config.symbols
        self._ibits = int(np.ceil(np.log2(self.Lin*self.B*self.Na)))
        self.ibits = config.index_bits
        self.sbits = config.symbol_bits
        self.rate = config.code_rate
        self.loss = {}
        self.keys = ['fer', 'mMSE', 'pMSE', 'pMSEf', 'pMSEm', 'pMSEL', 'ver', 'verf', 'verm', 'verL', 'ber', 'iber', 'sber', 'ier', 'ser']
        
        if config.is_complex:
            self.dtype = torch.complex64
            self.npdtype = np.complex64
        else:
            self.dtype = torch.float32
            self.npdtype = np.complex64
            
        if config.mode == 'segmented':
            self.section = self.Nt // self.Na
            self.decision = self.segmented_decision
        else:
            self.decision = self.random_decision

    def __call__(self, 
                 xamp: torch.Tensor, 
                 x: torch.Tensor,
                 symbols: np.ndarray = None,
                 indices: np.ndarray = None
                 ) -> None:
        """_summary_

        Args:
            xamp (torch.Tensor): _description_
            x (torch.Tensor): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        for key, value in zip(self.keys, self.error_rate(xamp, x, symbols, indices)):
            try:
                self.loss[key] = np.append(self.loss[key], value)
            except KeyError:
                self.loss[key] = np.array(value)
        
    def error_rate(self, 
                 xamp: torch.Tensor, 
                 x: torch.Tensor,
                 symbols: np.ndarray = None,
                 indices: np.ndarray = None
                 ) -> Tuple[np.ndarray]:
        """_summary_

        Args:
            xamp (torch.Tensor): _description_
            x (torch.Tensor): _description_
            symbols (np.ndarray): _description_
            indices (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        xamp, x = xamp.cpu().view(-1, self.Lin, self.Nt).numpy(), x.cpu().view(-1, self.Lin, self.Nt).numpy()
        xhat, shat, ihat = self.decision(xamp)
        xhat = xhat.reshape((-1, self.Lin, self.Nt))
        
        # measured mean Square Error
        mMSE = np.sum(np.abs(xamp - x)**2) / self.Ns
        
        # predicted mean square error
        pMSE, pMSEf, pMSEm, pMSEL = self.mean_square_error(xhat, x)
        
        # Section Error
        ver, verf, verm, verL = self.vector_error_rate(xhat, x)
        
        # Frame Error
        fer = self.frame_error_rate(xhat, x)
        
        # bit error rate, index error rate, symbol error rate
        ber, iber, sber, ier, ser = self.bit_error_rate(shat, ihat, symbols, indices)
        
        return fer, mMSE, pMSE, pMSEf, pMSEm, pMSEL, ver, verf, verm, verL, ber, iber, sber, ier, ser
    
    def mean_square_error(self, xhat: np.ndarray, x: np.ndarray):
        # Mean Square Error
        pMSE = np.sum(np.abs(xhat - x)**2) / self.Ns
        pMSEf = np.sum(np.abs(xhat[:, 0] - x[:, 0])**2) / self.Na / self.B
        pMSEm = np.sum(np.abs(xhat[:, self.Lin//2] - x[:, self.Lin//2])**2) / self.Na / self.B  
        pMSEL = np.sum(np.abs(xhat[:, -1] - x[:, -1])**2) / self.Na / self.B
        return pMSE, pMSEf, pMSEm, pMSEL
    
    def vector_error_rate(self, xhat: np.ndarray, x: np.ndarray):
        # vector Error
        ver = (np.count_nonzero(xhat.reshape((-1, self.Nt)) - x.reshape((-1, self.Nt)), axis=-1) > 0).sum() / self.Lin / self.B
        verf = (np.count_nonzero(xhat[:, 0] - x[:, 0], axis=-1) > 0).sum() / self.B
        verm = (np.count_nonzero(xhat[:, self.Lin//2] - x[:, self.Lin//2], axis=-1) > 0).sum() / self.B
        verL = (np.count_nonzero(xhat[:, -1] - x[:, -1], axis=-1) > 0).sum() / self.B
        return ver, verf, verm, verL
    
    def frame_error_rate(self, xhat: np.ndarray, x: np.ndarray):
        # Frame Error
        fer = (np.count_nonzero(xhat.reshape(self.B, -1) - x.reshape(self.B, -1), axis=-1) > 0).sum() / self.B
        return fer
    
    def bit_error_rate(self, sym_hat: np.ndarray, ind_hat: np.ndarray, sym: np.ndarray, ind: np.ndarray):
        ier = np.count_nonzero(ind_hat - ind) / self.Ns
        ser = np.count_nonzero(sym_hat - sym) / self.Ns
        # index bit error rate
        iber_ = np.count_nonzero(self.de2bi(np.bitwise_xor(ind_hat, ind), self._ibits)) / self.Lin / self.B
        iber = iber_ / self.ibits
        if self.sbits != 0:
            # symbol bit error rate
            sber_ = np.count_nonzero(self.de2bi(np.bitwise_xor(sym_hat, sym), self.sbits)) / self.Lin / self.B
            sber = sber_ / self.sbits / self.Na
        else:
            sber = 0.
            sber_ = 0.
        # bit error rate
        ber = (iber_ + sber_) / (self.Na * self.sbits + self.ibits)
        return ber, iber, sber, ier, ser
    
    def de2bi(self, dec: np.ndarray, bits: int):
        dec = dec.astype(int)
        bi = np.zeros((len(dec), bits), dtype=int)
        for i in range(bits):
            bi[:, i] = dec % 2
            dec = dec // 2
        return np.flip(bi, axis=1).ravel()
            
    def dumb_decision(self, xamp: np.ndarray) -> Tuple[np.ndarray]:
        xamp = xamp.ravel()
        index = np.sort(np.abs(xamp).argsort()[-self.Ns:])
        symbol = np.zeros_like(index)
        xhat = np.zeros_like(xamp)
        for j, xs in enumerate(xamp[index]):
            d = np.inf
            for i, s in enumerate(self.symbols):
                ds = np.abs(xs - s)
                if ds < d:
                    d = ds
                    symbol[j] = self.gray[i]
                    xhat[index[j]] = s
                    if ds == 0:
                        break
        return xhat, symbol, index
    
    # def segmented_decision(self, xamp: np.ndarray) -> Tuple[np.ndarray]:
    #     xamp = xamp.reshape(-1, self.Na)
    #     xamp2 = np.zeros_like(xamp)
    #     for j, x in enumerate(xamp):
    #         index = np.abs(x).argsort()[-1]
    #         xamp2[j, index] = x[index]
    #     xhat, symbol, index = self.random_decision(xamp2) 
    #     return xhat, symbol, index
    
    def segmented_decision(self, xamp: np.ndarray) -> Tuple[np.ndarray]:
        xamp = xamp.reshape(-1, self.Nt)
        xhat = np.zeros_like(xamp)
        xgray = np.zeros_like(xamp, dtype=int)
        for j, x in enumerate(xamp):
            index = [np.abs(x[self.section*k:self.section*(k+1)]).argsort()[-1] for k in range(self.Na)]
            xs = x[np.abs(x).argsort()[-1]]
            d = np.inf
            for i, s in enumerate(self.symbols):
                ds = np.abs(xs - s)
                if ds < d:
                    d = ds
                    xgray[j, index] = self.gray[i]
                    xhat[j, index] = s
                    if ds == 0:
                        break
        xhat = xhat.ravel()
        index = xhat.nonzero()[0]
        symbol = xgray.ravel()[index]
        return xhat, symbol, index
    
    def random_decision(self, xamp: np.ndarray) -> Tuple[np.ndarray]:
        xamp = xamp.reshape(-1, self.Nt)
        xhat = np.zeros_like(xamp)
        xgray = np.zeros_like(xamp, dtype=int)
        for j, x in enumerate(xamp):
            index = np.abs(x).argsort()[-self.Na:]
            xs = x[index[-1]]
            d = np.inf
            for i, s in enumerate(self.symbols):
                ds = np.abs(xs - s)
                if ds < d:
                    d = ds
                    xgray[j, index] = self.gray[i]
                    xhat[j, index] = s
                    if ds == 0:
                        break
        xhat = xhat.ravel()
        index = xhat.nonzero()[0]
        symbol = xgray.ravel()[index]
        return xhat, symbol, index
            
    def export(self, SNRdB: float, EbN0dB: float, save_location: str) -> None:

        self.loss['EbN0dB'] = float(EbN0dB)
        self.loss['SNRdB'] = float(SNRdB)
        self.loss['rate'] = float(self.rate)
        self.loss['C'] = float(np.log2(1 + 10**(SNRdB / 10)))
        
        for key in self.keys:
            self.loss[key] = list(self.loss[key])
        
        with open(f'{save_location}/{EbN0dB}.json', 'w', encoding='utf-8') as f:
            json.dump(self.loss, f, ensure_ascii=False, indent=6, skipkeys=True)
            
        self.loss = {}
    
    def accumulate(self, other):
        """_summary_

        Args:
            other (_type_): _description_
        """
        for key in self.keys:
            try:
                self.loss[key] += other.loss[key]
            except KeyError:
                self.loss[key] = other.loss[key]
            
    def average(self, epochs: int):
        """_summary_

        Args:
            epochs (int): _description_
        """
        for key in self.keys:
            self.loss[key] = list(np.array(self.loss[key]) / epochs)
            
    def dump(self):
        self.loss = {}