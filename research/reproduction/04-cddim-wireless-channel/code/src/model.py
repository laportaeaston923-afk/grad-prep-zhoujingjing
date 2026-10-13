"""Location-conditioned U-Net adapted from the architecture in taekyunl/cDDIM.

Input/output shape: [batch, 2 (Re,Im), 4, 32].
Fig. 2 uses base feature width 16; scale it via --n-feat for experiments.
"""
import torch
from torch import nn


class ResConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.same_channels = in_ch == out_ch
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(a)
        if self.residual:
            return (b + (x if self.same_channels else a)) / 1.4142135623730951
        return b


class Embed(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, out_features), nn.GELU(), nn.Linear(out_features, out_features))

    def forward(self, x):
        return self.net(x)


class ConditionalUNet(nn.Module):
    """Same core stages/conditioning rule as Fig. 2 and the official ContextUnet.

    Conditions: cemb * features + temb at the two upsampling resolutions.
    Note: no unused classifier-free context-mask or guidance paths are included.
    """
    def __init__(self, n_feat=16, in_channels=2, pos_dim=3):
        super().__init__()
        if n_feat % 8 != 0:
            raise ValueError('n_feat must be a multiple of 8 for GroupNorm')
        self.init = ResConv(in_channels, n_feat, residual=True)
        self.down1 = nn.Sequential(ResConv(n_feat, n_feat), nn.MaxPool2d(2))
        self.down2 = nn.Sequential(ResConv(n_feat, 2*n_feat), nn.MaxPool2d(2))
        self.to_vec = nn.Sequential(nn.AvgPool2d((1,8)), nn.GELU())
        self.time1 = Embed(1, 2*n_feat)
        self.time2 = Embed(1, n_feat)
        self.position1 = Embed(pos_dim, 2*n_feat)
        self.position2 = Embed(pos_dim, n_feat)
        self.up0 = nn.Sequential(nn.ConvTranspose2d(2*n_feat, 2*n_feat, (1,8), stride=(1,8)),
                                 nn.GroupNorm(8,2*n_feat), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(4*n_feat,n_feat,2,stride=2),ResConv(n_feat,n_feat),ResConv(n_feat,n_feat))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(2*n_feat,n_feat,2,stride=2),ResConv(n_feat,n_feat),ResConv(n_feat,n_feat))
        self.out = nn.Sequential(nn.Conv2d(2*n_feat,n_feat,3,padding=1),nn.GroupNorm(8,n_feat),nn.ReLU(),nn.Conv2d(n_feat,in_channels,3,padding=1))

    def forward(self, noisy, positions, t):
        if noisy.ndim != 4 or noisy.shape[1:] != (2,4,32):
            raise ValueError(f'Expected [B,2,4,32], got {list(noisy.shape)}')
        if t.ndim == 1: t = t[:,None]
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError('positions should be [B,3]')
        init = self.init(noisy)
        d1 = self.down1(init)
        d2 = self.down2(d1)
        bottleneck = self.up0(self.to_vec(d2))
        p1 = self.position1(positions).unsqueeze(-1).unsqueeze(-1)
        p2 = self.position2(positions).unsqueeze(-1).unsqueeze(-1)
        t1 = self.time1(t).unsqueeze(-1).unsqueeze(-1)
        t2 = self.time2(t).unsqueeze(-1).unsqueeze(-1)
        u1 = self.up1(torch.cat((p1*bottleneck+t1, d2), dim=1))
        u2 = self.up2(torch.cat((p2*u1+t2,d1),dim=1))
        return self.out(torch.cat((u2,init),dim=1))
