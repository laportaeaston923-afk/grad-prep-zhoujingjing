"""Render actual 1/5-step sampling states from a raw MeanFlow state_dict."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from models.dit import MFDiT, Attention
from repro_utils import load_config, build_flow, generate, environment


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def sample_states(model, labels, noise, steps, cfg_scale):
    z = noise.clone()
    states = [z.detach().cpu().clone()]
    w = torch.full((len(labels),), cfg_scale, device=z.device)
    for i in range(steps):
        t = torch.full((len(labels),), 1 - i / steps, device=z.device)
        r = torch.full((len(labels),), 1 - (i + 1) / steps, device=z.device)
        u, _ = model(z, t, r, y=labels, w=w)
        z = z - u / steps
        if not torch.isfinite(z).all():
            raise ValueError('Non-finite sampling state')
        states.append(z.detach().cpu().clone())
    return states


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def render_panel(state, flow, labels, steps, update):
    # Clipping is for display only. The raw state goes unchanged to the next update.
    pixels = flow.normer.unnorm(state).float().clamp(0, 1).mul(255).round().to(torch.uint8)
    if pixels.shape[1] != 1:
        raise ValueError('This visualizer is for grayscale MNIST checkpoints')
    cell, gap, margin = 80, 10, 20
    width = margin * 2 + 5 * cell + 4 * gap
    panel = Image.new('RGB', (width, 300), 'white')
    draw = ImageDraw.Draw(panel)
    caption = 'Initial noise' if update == 0 else ('Final image' if update == steps else 'Intermediate state')
    draw.text((margin, 12), f'{steps}-step | update {update}/{steps} | t={1-update/steps:.1f}', fill='#182838', font=font(19))
    draw.text((margin, 40), caption, fill='#425466', font=font(15))
    for index, label in enumerate(labels.tolist()):
        row, col = divmod(index, 5)
        x, y = margin + col * (cell + gap), 76 + row * 110
        image = Image.fromarray(pixels[index, 0].numpy()).resize((cell, cell), Image.Resampling.NEAREST)
        panel.paste(image.convert('RGB'), (x, y))
        draw.text((x + 24, y + cell + 3), f'y={label}', fill='#182838', font=font(15))
    return panel


def save_storyboard(panels, path, columns):
    width, height = panels[0].size
    rows = math.ceil(len(panels) / columns)
    sheet = Image.new('RGB', (columns * (width + 12) + 12, rows * (height + 12) + 12), '#dce3eb')
    for index, panel in enumerate(panels):
        row, col = divmod(index, columns)
        sheet.paste(panel, (12 + col * (width + 12), 12 + row * (height + 12)))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cfg-scale', type=float, default=1.0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    cfg = load_config(args.config)
    if cfg.get('dataset') != 'mnist' or cfg['model']['num_classes'] != 10:
        parser.error('Use an MNIST checkpoint with ten classes')
    scale = cfg['meanflow'].get('cfg_scale')
    supported = args.cfg_scale == 1 or (isinstance(scale, (list, tuple)) and scale[0] <= args.cfg_scale <= scale[1]) or (isinstance(scale, (int, float)) and args.cfg_scale == scale)
    if not supported:
        parser.error('CFG scale is outside the configured training range')
    device = torch.device('cpu' if args.cpu else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --cpu')
    torch.set_num_threads(args.threads)
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MFDiT(**cfg['model']).to(device).eval()
    model.load_state_dict(weights, strict=True)
    import torch.nn.functional as functional
    Attention._native_attention = lambda self, q, k, v: functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
    flow = build_flow(cfg)
    labels = torch.arange(10, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(10, 1, cfg['model']['input_size'], cfg['model']['input_size'], device=device, generator=rng)
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = dict(checkpoint_filename=args.checkpoint.name,
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    config=cfg, config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    seed=args.seed, cfg_scale=args.cfg_scale, labels=labels.tolist(),
                    initial_noise_sha256=tensor_hash(noise), generation_precision='FP32',
                    attention_backend='sdpa', environment=environment(device),
                    display='unnorm every raw state, clip [0,1], round(255*x), nearest-neighbor display enlargement',
                    note='Ten fixed samples for process illustration, not the 10000-image FID evaluation; '
                         'GIF repeats recorded states for viewing and adds no interpolated model states.',
                    results=[])
    for steps in [1, 5]:
        count = [0]
        hook = model.register_forward_hook(lambda *unused: count.__setitem__(0, count[0] + 1))
        states = sample_states(model, labels, noise, steps, args.cfg_scale)
        hook.remove()
        assert count[0] == steps
        expected = generate(model, flow, labels, noise, steps, args.cfg_scale)
        actual = flow.normer.unnorm(states[-1].to(device))
        error = float((expected - actual).abs().max())
        assert error < 1e-6, error
        panels = [render_panel(state, flow, labels.cpu(), steps, i) for i, state in enumerate(states)]
        save_storyboard(panels, args.out / f'process_{steps}step.png', columns=2 if steps == 1 else 3)
        durations = ([1300, 2300] if steps == 1 else [1000] + [650] * 4 + [2300])
        panels[0].save(args.out / f'process_{steps}step.gif', save_all=True,
                       append_images=panels[1:], duration=durations, loop=0, disposal=2, optimize=False)
        protocol['results'].append(dict(sample_steps=steps, nfe=count[0], recorded_states=len(states),
            times=[1-i/steps for i in range(steps+1)], state_sha256=[tensor_hash(s) for s in states],
            final_image_max_abs_difference_vs_generate=error))
        print(f'{steps}-step: {len(states)} states; {count[0]} model calls; final verification error {error}')
    (args.out / 'process.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
