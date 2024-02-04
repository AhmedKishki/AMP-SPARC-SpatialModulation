import os

import json
import torch
from torch import nn
import numpy as np

from config import Config
from data import Data
from channel import Channel
from loss import Loss
from vamp import VAMP
from plotter import Plotter


class Model:
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        # build
        self.rate = config.code_rate
        self.shannon_limit = config.shannon_limit_dB
        self.min_snr = self.shannon_limit
        self.amp = VAMP(config).to(config.device)
        self.loss = Loss(config)
        self.channel = Channel(config)
        self.data = Data(config)
        self.path = f'Simulations/VAMP/{config.name}'
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
    
    @torch.no_grad()
    def simulate(self, epochs: int, final = None, start = None, step: float = 1, res: int = 1):
        if start is None:
            start = int(np.ceil(self.min_snr))
        if final is None:
            final = start + 10.0
        EbN0dB_range = np.arange(start, final+step, step)
        SNRdB_range = EbN0dB_range + 10*np.log10(self.rate)
        for SNRdB, EbN0dB in zip(SNRdB_range, EbN0dB_range):
            print(f'EbN0dB={EbN0dB}')
            SNR = 10 ** ( SNRdB / 10)
            for i in range(epochs):
                if i % res == 0:
                    H = self.channel.generate_channel()
                    U, s, Vh = torch.linalg.svd(H, full_matrices=False)
                x, sym, idx = self.data.generate_message()
                y = H @ x + self.channel.awgn(SNR)
                loss = self.amp(U, s, Vh, y, SNR, x, sym, idx)
                self.loss.accumulate(loss)
            self.loss.average(epochs)
            print(f"FER={self.loss.loss['fer']}, iter={self.loss.loss['T']}")
            self.loss.export(SNRdB, EbN0dB, self.path)

if __name__ == "__main__":
    alph = 'OOK'
    # Nt = 1344
    # Na = 84
    # Nr = 73
    # Lin = 32
    # Lh = 6
    Nt = 128
    Na = 8
    Nr = 24
    Lin = 20
    Lh = 3
    for trunc in ['trunc']:
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
                                iterations=200
                                )
                print(config.__dict__)
                model = Model(config)
                model.simulate(epochs=100, step=1, start=11.0, final=15.0, res=100)
                Plotter(config, 'VAMP').plot_iter()
                Plotter(config, 'VAMP').plot_metrics()