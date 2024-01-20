import os

import json
import torch
from torch import nn
import numpy as np

from config import Config
from data import Data
from channel import Channel
from loss import Loss
from bamp import BAMP
from plotter import Plotter


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # build
        self.rate = config.code_rate
        self.BAMP = BAMP(config).to(config.device)
        self.loss = Loss(config)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = f'Simulations/RANDOM/{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
        # with open(f'{self.path}/config.json', 'w', encoding='utf-8') as f:
        #     json.dump(config.__dict__, f, ensure_ascii=F  alse, indent=6, skipkeys=True)

    @torch.no_grad()
    def run(self, SNR: float) -> Loss:
        x, s, i = self.data.generate_message()
        H = self.channel.generate_as_random()
        y = H @ x + self.channel.awgn(SNR)
        loss = self.AMP(x, y, H, SNR, s, i)
        print(loss.loss)
        return loss
    
    def simulate(self, epochs: int, step: float = 1, initial: float=0., final: float=10.):
        EbN0 = np.arange(initial, final+step, step)
        SNRdB_range = EbN0 + 10*np.log10(self.rate)
        for SNRdB, EbN0dB in zip(SNRdB_range, EbN0):
            SNR = 10 ** ( SNRdB / 10)
            for i in range(epochs):
                print(EbN0dB, i)
                self.loss.accumulate(self.run(SNR))
            self.loss.average(epochs)
            print(self.loss.loss)
            self.loss.export(SNRdB, EbN0dB, self.path)

if __name__ == "__main__":
    Nt = 128
    Nr = 32
    Lin = 20
    for trunc in ['tail']:
        for Lh in [3]:
        # for Lh in [5]:
            for Na in [1, 2, 4]: 
                for alph in ['QPSK', '8PSK','16PSK']:
                # for alph in ['16QAM']:
                    for prof in ['uniform']:
                    # for prof in ['exponential']:
                        for gen in ['sparc']:
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
                                            batch=100
                                            )
                            print(config.__dict__)
                            Model(config).simulate(epochs=100, step=0.5)
                            Plotter(config, 'RANDOM').plot_metrics()