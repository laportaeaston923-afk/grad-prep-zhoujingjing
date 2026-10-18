"""Conditional DDPM for simulated CSI amplitude recovery (no downloaded weights)."""
import math
import numpy as np
import torch
from torch import nn

CHANNELS, LENGTH, FS, STEPS = 8, 128, 10.0, 100


def simulate(count, seed):
    """N independent multipath windows. Returns clean, observed, mask, motion Hz.

    H_k(t)=sum_l a_l exp(-j 2pi f_k tau_l(t)); only one path moves.
    Static per-subcarrier means are removed, using clean calibration statistics.
    This ideal calibration is a simulation assumption, not a real CSI pipeline.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(LENGTH) / FS
    carriers = 5.32e9 + np.arange(CHANNELS) * 312500
    clean, observed, masks, rates = [], [], [], []
    for _ in range(count):
        frequency = rng.uniform(.18, .42) if rng.random() < .5 else rng.uniform(.9, 1.6)
        motion = rng.uniform(.0015, .006) * np.sin(2*np.pi*frequency*time + rng.uniform(0, 2*np.pi))
        distances = rng.uniform(2, 12, 3)
        gains = np.array([1., .5, rng.uniform(.25, .5)]) * np.exp(1j*rng.uniform(0, 2*np.pi, 3))
        h = np.zeros((CHANNELS, LENGTH), dtype=complex)
        for i in range(3):
            d = distances[i] + (motion if i == 2 else np.zeros(LENGTH))
            h += gains[i] * np.exp(-2j*np.pi*carriers[:, None]*d[None, :]/3e8)
        x = np.abs(h)
        x = (x - x.mean(axis=1, keepdims=True)) / .25
        sigma = rng.uniform(.15, .45)
        mask = np.broadcast_to(rng.random((1, LENGTH)) > rng.uniform(.05, .2), x.shape).copy()
        y = (x + rng.normal(0, sigma, x.shape)) * mask
        clean.append(x); observed.append(y); masks.append(mask); rates.append(frequency)
    return tuple(np.asarray(a, dtype=np.float32) for a in (clean, observed, masks, rates))


class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                                 nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, width, 1))
        self.time = nn.Linear(width, width)

    def forward(self, x, t):
        return (x + self.net(x + self.time(t)[:, :, None])) / math.sqrt(2)


class Denoiser(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.register_buffer('freq', torch.exp(-math.log(10000)*torch.arange(width//2)/(width//2-1)))
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.input = nn.Conv1d(CHANNELS*3, width, 3, padding=1)
        self.blocks = nn.ModuleList([Block(width, d) for d in (1, 2, 4, 8)])
        self.output = nn.Conv1d(width, CHANNELS, 3, padding=1)

    def forward(self, x, t, observed, mask):
        angle = t[:, None] * self.freq[None, :]
        embedding = self.time(torch.cat([angle.sin(), angle.cos()], dim=1))
        z = self.input(torch.cat([x, observed, mask], dim=1))
        for block in self.blocks:
            z = block(z, embedding)
        return self.output(z)


def schedule(device='cpu'):
    f = torch.cos((torch.linspace(0, 1, STEPS+1, device=device)+.008)/1.008*math.pi/2)**2
    beta = (1-f[1:]/f[:-1]).clamp(max=.999)
    alpha = 1-beta
    abar = alpha.cumprod(0)
    previous = torch.cat([torch.ones(1, device=device), abar[:-1]])
    posterior_variance = beta * (1-previous)/(1-abar)
    return beta, alpha, abar, posterior_variance


@torch.no_grad()
def sample(model, observed, mask, seed=2026, keep_frames=False):
    """Full 100-step ancestral DDPM; independent seeded sampling generator."""
    model.eval()
    device = observed.device
    generator = torch.Generator(device=device).manual_seed(seed)
    beta, alpha, abar, variance = schedule(device)
    x = torch.randn(observed.shape, generator=generator, device=device)
    frames = [(STEPS, x.detach().cpu().numpy())] if keep_frames else []
    for i in reversed(range(STEPS)):
        t = torch.full((len(x),), i, device=device, dtype=torch.long)
        eps = model(x, t, observed, mask)
        # Clipped x0 prediction and exact posterior mean.
        x0 = ((x-(1-abar[i]).sqrt()*eps)/abar[i].sqrt()).clamp(-2, 2)
        prev = abar[i-1] if i else torch.ones((), device=device)
        mean = beta[i]*prev.sqrt()/(1-abar[i])*x0 + (1-prev)*alpha[i].sqrt()/(1-abar[i])*x
        x = mean + (variance[i].sqrt()*torch.randn(x.shape, generator=generator, device=device) if i else 0)
        if keep_frames and i % 5 == 0:
            frames.append((i, x.detach().cpu().numpy()))
    return x, frames
