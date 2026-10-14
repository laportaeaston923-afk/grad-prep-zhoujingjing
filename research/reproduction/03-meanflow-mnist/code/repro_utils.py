"""Single-device utilities; upstream model/loss are kept unchanged."""
import copy
import importlib.util
import json
import platform
from pathlib import Path

import torch

UPSTREAM_COMMIT = '97cc5182abe9d69e5dcb350268f603fbe31a5897'


def load_config(path):
    spec = importlib.util.spec_from_file_location('repro_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return copy.deepcopy(module.config)


def build_flow(cfg):
    if 'comparison' in cfg:
        from comparison_flow import MFIMFComparison
        kwargs = dict(cfg['comparison'])
        kwargs.update(channels=cfg['model']['in_channels'],
                      image_size=cfg['model']['input_size'],
                      num_classes=cfg['model']['num_classes'])
        return MFIMFComparison(**kwargs)
    from meanflow import MeanFlow
    kwargs = dict(cfg['meanflow'])
    kwargs.update(channels=cfg['model']['in_channels'],
                  image_size=cfg['model']['input_size'],
                  num_classes=cfg['model']['num_classes'])
    return MeanFlow(**kwargs)


def device_and_dtype(precision='bf16', cpu=False):
    device = torch.device('cpu' if cpu else 'cuda')
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA PyTorch, or use --cpu for a smoke test.')
    if device.type == 'cpu' or precision == 'no':
        return device, None
    if precision != 'bf16':
        raise ValueError('This trainer supports bf16 or no; use --precision no for FP32.')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unavailable; use --precision no.')
    return device, torch.bfloat16


def autocast(device, dtype):
    return torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None)


def environment(device):
    import importlib.metadata
    packages = {}
    for name in ['torch', 'torchvision', 'timm', 'einops', 'numpy']:
        packages[name] = importlib.metadata.version(name)
    return dict(python=platform.python_version(), packages=packages,
                cuda_build=torch.version.cuda, device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                upstream_commit=UPSTREAM_COMMIT)


@torch.no_grad()
def generate(model, flow, labels, noise, n_steps=1, cfg_scale=1.0):
    if n_steps < 1:
        raise ValueError('n_steps must be positive')
    z = noise.clone()
    w = torch.full((len(labels),), cfg_scale, device=z.device)
    for i in range(n_steps):
        t = torch.full((len(labels),), 1.0 - i / n_steps, device=z.device)
        r = torch.full((len(labels),), 1.0 - (i + 1) / n_steps, device=z.device)
        u, _ = model(z, t, r, y=labels, w=w)
        z = z - u / n_steps
    return flow.normer.unnorm(z)


def plot_history(rows, target):
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    steps = [row['step'] for row in rows]
    axes[0].plot(steps, [row['loss'] for row in rows], label='Total adaptive loss')
    axes[1].plot(steps, [row['mf_v_mse'] for row in rows], label='Raw MF velocity MSE')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    axes[1].set_xlabel('Training step (real interval averages)')
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
