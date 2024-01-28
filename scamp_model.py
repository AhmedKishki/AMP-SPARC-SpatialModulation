import os

import json
import torch
from torch import nn
import numpy as np

from config import Config
from data import Data
from channel import Channel
from loss import Loss
from scamp import SCAMP
from plotter import Plotter


class Model(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # build
        self.rate = config.code_rate
        self.shannon_limit = config.shannon_limit_dB
        self.min_snr = self.shannon_limit
        self.amp = SCAMP(config).to(config.device)
        self.loss = Loss(config)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = f'Simulations/SCAMP/{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
        # with open(f'{self.path}/config.json', 'w', encoding='utf-8') as f:
        #     json.dump(config.__dict__, f, ensure_ascii=F  alse, indent=6, skipkeys=True)

    @torch.no_grad()
    def run(self, SNR: float) -> Loss:
        x, s, i = self.data.generate_message()
        W, A = self.channel.generate_as_sparc()
        y = A @ x + self.channel.awgn(SNR)
        loss = self.amp(W, A, y, SNR, x, s, i)
        print(loss.loss['fer'], loss.loss['T'])
        return loss
    
    def simulate(self, epochs: int, final = 10, start = 0, step: float = 1, res:int=1):
        start = int(np.floor(self.min_snr + start))
        EbN0dB_range = np.arange(start, final+step, step)
        SNRdB_range = EbN0dB_range + 10*np.log10(self.rate)
        for SNRdB, EbN0dB in zip(SNRdB_range, EbN0dB_range):
            SNR = 10 ** ( SNRdB / 10)
            for i in range(epochs):
                if i % res == 0:
                    W, A = self.channel.generate_as_sparc()
                x, s, i = self.data.generate_message()
                y = A @ x + self.channel.awgn(SNR)
                loss = self.amp(W, A, y, SNR, x, s, i)
                self.loss.accumulate(loss)
            self.loss.average(epochs)
            print(f'EbN0dB={EbN0dB}')
            print(f"FER={self.loss.loss['fer']}, iter={self.loss.loss['T']}")
            self.loss.export(SNRdB, EbN0dB, self.path)

if __name__ == "__main__":
    alph = 'OOK'
    Nt = 1344
    Na = 84
    Nr = 73
    Lin = 32
    Lh = 6
    # Nt = 128
    # Na = 8
    # Nr = 24
    # Lin = 20
    # Lh = 3
    for trunc in ['tail']:
        for prof in ['uniform']:
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
                                batch=1,
                                iterations=200
                                )
                print(config.__dict__)
                model = Model(config)
                # model.simulate(epochs=1000, step=1, final=6.0, res=100)
                Plotter(config, 'SCAMP').plot_iter()
                Plotter(config, 'SCAMP').plot_metrics()