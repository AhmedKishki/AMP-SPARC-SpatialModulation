import os

import numpy as np
import pandas as pd
import torch

from config import Config
from channel import Channel
from data import Data

from info_theory import mi_awgn

class InfoTheory:
    def __init__(self, config: Config) -> None:
        config.device = 'cpu'
        self.Nt, self.Na, self.Nr = config.Nt, config.Na, config.Nr
        self.symbols = config.symbols
        self.ps = config.Ps
        self.channel = Channel(config)
        self.rate = config.code_rate
        self.shannon_limit = config.shannon_limit_dB
        self.min_snr = self.shannon_limit
        self.path = f'Simulations/Capacity/{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
    def simulate(self, epochs: int = 1000, final = None, start = None, step: float = 1):
        if start is None:
            start = int(np.ceil(self.min_snr))
        if final is None:
            final = start + 10.0
        EbN0dB_range = np.arange(start, final+step, step)
        SNRdB_range = EbN0dB_range + 10*np.log10(self.rate)
        capacity = []
        for SNRdB, EbN0dB in zip(SNRdB_range, EbN0dB_range):
            print(f'EbN0dB={EbN0dB}')
            SNR = 10 ** ( SNRdB / 10)
            sigma2 = 1 / SNR
            Cawgn = np.log2(1 + SNR)
            Cwf = 0.0
            Cfs = 0.0
            Mi = 0.0
            for _ in range(epochs):
                H = self.channel.generate_channel()
                g = torch.linalg.svdvals(H).numpy()**2
                Pwf = self._water_filling(g, sigma2)
                mi = self._mutual_information(g, sigma2)
                Cwf = np.max([Cwf, np.sum(np.log2(1 + g * Pwf / sigma2))])
                # Cfs = np.max([Cfs, np.sum(np.log2(1 + g / sigma2 / len(g)))])
                Mi = np.max([Mi, mi])
                print(mi, Cawgn)
            capacity.append([Cawgn, Cfs, Cwf, mi])
            print(f'Cawgn: {Cawgn}, Cwf: {Cwf}, Cfs: {Cfs}, Mi: {Mi}')
        capacity = np.array(capacity)
        Cdict = {'EbN0dB': EbN0dB_range, 'SNRdB': SNRdB_range, 'Cawgn': capacity[:, 0], 'Cfs': capacity[:, 1], 'Cwf': capacity[:, 2], 'Mi': capacity[:, 3]}
        pd.DataFrame(Cdict).to_csv(f'{self.path}/{config.Nt,config.Na,config.Nr,config.Lh}.csv')
        return capacity
    
    def _water_filling(self, gain: np.ndarray, sigma2: float, power: float = 1) -> np.ndarray:
        """
        Calculates the water level that touches the worst channel (the higher
        one) and therefore transmits zero power in this worst channel. After
        that, calculates the power in each channel (the vector 'Ps') for this
        water level. If the sum of all of these powers in 'Ps' is less then
        the total available power, then all we need to do is divide the
        remaining power equally among all the channels (increase the water
        level). On the other hand, if the sum of all of these powers in 'Ps'
        is greater then the total available power then we remove the worst
        channel and repeat the process. 
        Calculates minimum water-level $\mu$ required to use all channels

        Args:
            gain (np.ndarray): _description_
            sigma2 (float): _description_

        Returns:
            _type_: _description_
        """
        P = power
        gain = gain * self.Nr / self.Nt
        NChannels = len(gain)
        RemoveChannels = 0

        minMu = sigma2 / (gain[NChannels - RemoveChannels - 1])
        Ps = (minMu - sigma2 /(gain[:NChannels - RemoveChannels]))

        while (sum(Ps) > P) and (RemoveChannels < NChannels):
            RemoveChannels += 1
            minMu = sigma2 / (gain[NChannels - RemoveChannels - 1])
            Ps = minMu - sigma2 / (gain[:NChannels - RemoveChannels])

        # Distributes the remaining power among the all the remaining channels
        Pdiff = P - np.sum(Ps)
        Paux = Pdiff / (NChannels - RemoveChannels) + Ps

        # Put optimum power in the original channel order
        Palloc = np.zeros(NChannels)
        Palloc[:NChannels - RemoveChannels] = Paux

        return Palloc
    
    def _mutual_information(self, gain: np.ndarray, SNR: float, N: int = 100) -> float:
        gain = gain / len(gain)
        symbols = self.symbols
        
        ps = self.ps
        x = np.append(symbols, 0)
        pmf_x = np.ones_like(symbols, dtype=np.float32) * ps
        pmf_x = np.append(pmf_x, 1 - np.sum(pmf_x))
        
        Px = np.sum(np.abs(x)**2 * pmf_x)
        sigma2 = Px / SNR
        
        xmax = np.amax(np.abs(x))
        ymax = xmax + 10 * np.sqrt(sigma2)
        ygrid = np.linspace(-ymax, ymax, N)
        yr, yi = np.meshgrid(ygrid, ygrid)
        y = (yr + 1j*yi).flatten()
        
        mi = 0
        for g in gain:
            pmf_y = np.zeros_like(y, dtype=np.float32)
            pmf_y_x = np.zeros((len(y), len(x)), dtype=np.float32)
            log_pmf_y = np.zeros_like(pmf_y)
            log_pmf_y_x = np.zeros_like(pmf_y_x)
            
            for i, s in enumerate(x):
                tmp = np.exp(- np.abs(y - np.sqrt(g)*s)**2 / sigma2)
                pmf_y_x[:, i] = tmp / np.sum(tmp)
                pmf_y += pmf_y_x[:, i] * pmf_x[i]
            
            log_pmf_y_x[np.nonzero(pmf_y_x)] = np.log2(pmf_y_x[np.nonzero(pmf_y_x)])
            log_pmf_y[np.nonzero(pmf_y)] = np.log2(pmf_y[np.nonzero(pmf_y)])
            
            for i, _ in enumerate(x):
                mi += np.sum(pmf_y_x[:, i] * (log_pmf_y_x[:, i] - log_pmf_y)) * pmf_x[i]
            
        return mi
        
if __name__ == "__main__":
    Nt = 128
    Lin = 20
    Lh = 3
    alph = 'QPSK'
    for trunc in ['tail']:
        for Nr in [32, 24]:
            for Na in [4, 8]:
                for prof in ['uniform']:
                    for gen in ['segmented']:
                        config = Config(
                                        N_transmit_antenna=Nt,
                                        N_active_antenna=Na,
                                        N_receive_antenna=Nr,
                                        block_length=Lin,
                                        channel_length=Lh,
                                        channel_truncation=trunc,
                                        alphabet=alph,
                                        channel_profile=prof,
                                        generator_mode=gen,
                                        batch=1,
                                        iterations=50
                                        )
                        print(config.__dict__)
                        print(InfoTheory(config).simulate(epochs=100))
    