import torch
from torch import nn

from config import Config
from data import GSSK
from channel import Channel
from vamp import VAMP

class Model:
    def __init__(self, config: Config) -> None:
        # build
        self.device = config.device
        self.sparsity = config.sparsity
        self.vamp = VAMP(config).to(config.device)
        self.channel = Channel(config)
        self.data = GSSK(config)
        
        # error calculation
        self.MSE = nn.MSELoss()
        self.error_rate = lambda x: torch.count_nonzero(x) / x.numel()

    @torch.no_grad()
    def run(self, SNR_db: float):
        sigma2_N = 0.5 * 10 ** (SNR_db / 10)
        self.channel.generate(svd=True)
        x = self.data.generate()
        y = self.channel(x, sigma2_N)
        xhat, loss, error_rate = self.vamp(x, y, self.channel)
        return x, xhat, loss, error_rate
    
    def simulate(self, runs, SNR_db_start, SNR_db_final, SNR_db_step):
        pass
        
if __name__ == "__main__":
    config = Config(batch=1, 
                    N_transmit_antenna=100,
                    N_active_antenna=5,
                    N_receive_antenna=50, 
                    block_length=20,
                    channel_length=15,
                    iterations=10, 
                    alphabet='BPSK')
    model = Model(config)
    x, xhat, mse, error = model.run(10.)
    print(error)