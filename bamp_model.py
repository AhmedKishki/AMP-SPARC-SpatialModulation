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
        self.shannon_limit = config.shannon_limit_dB
        self.min_snr = int(np.floor(self.shannon_limit))
        self.amp = BAMP(config).to(config.device)
        self.loss = Loss(config)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = f'Simulations/BAMP/{config.name}'
        os.makedirs(self.path, exist_ok=True)
        
        # with open(f'{self.path}/config.json', 'w', encoding='utf-8') as f:
        #     json.dump(config.__dict__, f, ensure_ascii=F  alse, indent=6, skipkeys=True)

    @torch.no_grad()
    def run(self, SNR: float) -> Loss:
        x, s, i = self.data.generate_message()
        H = self.channel.generate_channel()
        y = H @ x + self.channel.awgn(SNR)
        loss = self.amp(x, y, H, SNR, s, i)
        print(loss.loss)
        return loss
    
    def simulate(self, epochs: int, final = 10, start = None, step: float = 1):
        if start is None:
            start = self.min_snr
        EbN0dB_range = np.arange(start, final+step, step)
        SNRdB_range = EbN0dB_range + 10*np.log10(self.rate)
        for SNRdB, EbN0dB in zip(SNRdB_range, EbN0dB_range):
            SNR = 10 ** ( SNRdB / 10)
            for i in range(epochs):
                print(f'EbN0dB={EbN0dB}, epoch={i}')
                if i % 20 == 0:
                    H = self.channel.generate_channel()
                x, s, i = self.data.generate_message()
                y = H @ x + self.channel.awgn(SNR)
                loss = self.amp(H, y, SNR, x, s, i)
                self.loss.accumulate(loss)
                print(f"FER={loss.loss['fer']}, iter={loss.loss['T']}")
            self.loss.average(epochs)
            print(self.loss.loss)
            self.loss.export(SNRdB, EbN0dB, self.path)

if __name__ == "__main__":
    Na = 84
    Nr = 73
    Lin = 32
    for trunc in ['tail']:
        for Lh in [6]:
            for (Nt, alph) in [(1344,'OOK'),(672,'BPSK'),(336,'QPSK'),(168,'8PSK')]:
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
                                        iterations=50
                                        )
                        print(config.__dict__)
                        Model(config).simulate(epochs=200, step=0.25)
                        Plotter(config, 'BAMP').plot_metrics()