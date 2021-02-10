from piqa import ssim

class SSIMLoss(ssim.SSIM):
    def __init__(self, device):
        super().__init__()
        self.device = device
        print('SSIMLoss', self.device)
    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        return 1 - super().forward(x, y)