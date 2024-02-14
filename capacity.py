import os

import numpy as np
import pandas as pd
import torch

from config import Config
from channel import Channel
from data import Data

from infotheory import mi_awgn

class Capacity:
    def __init__(self, config: Config) -> None:
        config.device = 'cpu'
        self.Nt, self.Na, self.Nr = config.Nt, config.Na, config.Nr
        self.N = config.Nt * config.Lin
        self.n = config.Nr * config.Lout
        self.channel = Channel(config)
        self.rate = config.code_rate
        self.shannon_limit = config.shannon_limit_dB
        self.min_snr = self.shannon_limit
        self.path = f'Simulations/Capacity/{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
    @torch.no_grad()
    def calculate(self, epochs: int, final = None, start = None, step: float = 1):
        if start is None:
            start = int(np.ceil(self.min_snr))
        if final is None:
            final = start + 10.0
        EbN0dB_range = np.arange(start, final+step, step)
        SNRdB_range = EbN0dB_range + 10*np.log10(self.rate)
        x = np.ones(self.n)*np.sqrt(self.Na / self.Nt)
        capacity = []
        for SNRdB, EbN0dB in zip(SNRdB_range, EbN0dB_range):
            print(f'EbN0dB={EbN0dB}')
            SNR = 10 ** ( SNRdB / 10)
            sigma2 = self.Na / self.Nr / SNR
            Cawgn = np.log2(1 + SNR)
            Cfs = 0.0
            for _ in range(epochs):
                H = self.channel.generate_channel()
                g = torch.linalg.svdvals(H).numpy()**2
                Pwf = self.water_filling(g, sigma2)
                print(self.Na / self.Nr)
                print(np.sum(Pwf*g))
                Cwf = np.max([Cfs, np.sum(np.log2(1 + g * Pwf / sigma2))])
            capacity.append([Cawgn, Cwf])
            print(f'Cawgn: {Cawgn}, Cwf: {Cwf}')
        capacity = np.array(capacity)
        Cdict = {'EbN0dB': EbN0dB_range, 'SNRdB': SNRdB_range, 'Cawgn': capacity[:, 0], 'Cwf': capacity[:, 1]}
        pd.DataFrame(Cdict).to_csv(f'{self.path}/{config.Nt,config.Na,config.Nr,config.Lh}.csv')
        return capacity
    
    def water_filling(self, gain: np.ndarray, sigma2: float) -> np.ndarray:
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
        P = self.Na / self.Nt
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

if __name__ == "__main__":
    Nt = 128
    Lin = 20
    Lh = 3
    alph='OOK'
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
                        print(Capacity(config).calculate(epochs=100))
    