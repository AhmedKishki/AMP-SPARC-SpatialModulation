import torch
from torch import nn

from config import Config
from data import GSSK
from channel import Channel
from amp import AMP

class Model:
    def __init__(self, config: Config) -> None:
        # build
        self.device = config.device
        self.sparsity = config.sparsity
        self.AMP = AMP(config).to(config.device)
        self.channel = Channel(config)
        self.data = GSSK(config)
        
        # error calculation
        self.MSE = nn.MSELoss()
        self.error_rate = lambda x: torch.count_nonzero(x) / x.numel()

    @torch.no_grad()
    def run(self, SNR_db: float):
        sigma2_N = 0.5 * 10 ** ( - SNR_db / 10)
        print(sigma2_N)
        self.channel.generate()
        x = self.data.generate()
        y = self.channel(x, sigma2_N)
        xhat, mse, ber = self.AMP(x, y, self.channel, sigma2_N)
        return xhat, mse, ber
    
    def simulate(self, runs, SNR_db_start, SNR_db_final, SNR_db_step):
        pass
        
if __name__ == "__main__":
    config = Config(2, 100, 5, 50, 10, 5, 100)
    model = Model(config)
    xhat, mse, ber = model.run(0.)
    print(ber)