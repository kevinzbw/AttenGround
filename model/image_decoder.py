import torch
import torch.nn as nn
import torchvision

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class ImageDecoder(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.deconv = nn.Sequential(
            # hidden_dim//2, 34, 34
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(True),
            # hidden_dim//4, 68, 68
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim//4),
            nn.ReLU(True),
            # hidden_dim//4, 136, 136
            nn.ConvTranspose2d(hidden_dim//4, hidden_dim//20, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim//20),
            nn.ReLU(True),
            # hidden_dim//4, 272, 272
            nn.ConvTranspose2d(hidden_dim//20, 3, kernel_size=4, stride=2, padding=1, bias=False),
            Interpolate(size=(output_dim, output_dim), mode='bilinear'),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.deconv(x)