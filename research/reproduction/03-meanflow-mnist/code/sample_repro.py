"""Generate and time 1/5-step samples from a train_repro checkpoint."""
import argparse
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from torchvision.utils import save_image

from models.dit import MFDiT
from repro_utils import autocast, build_flow, device_and_dtype, environment, generate, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--sample-steps', type=int, nargs='+', default=[1, 5])
    p.add_argument('--n-per-class', type=int, default=10)
    p.add_argument('--cfg-scale', type=float, default=1.0)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision', choices=['bf16', 'no'])
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if min(args.sample_steps + [args.n_per_class, args.repeats]) < 1:
        p.error('Steps, sample counts and repeats must be positive')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    scale = cfg['meanflow'].get('cfg_scale')
    if scale is None and args.cfg_scale != 1.0:
        p.error('This checkpoint was trained without CFG distillation; use --cfg-scale 1')
    if isinstance(scale, (int, float)) and args.cfg_scale not in [1.0, float(scale)]:
        p.error('CFG scale was not trained')
    if isinstance(scale, (list, tuple)) and not (args.cfg_scale == 1 or scale[0] <= args.cfg_scale <= scale[1]):
        p.error('CFG scale is outside the trained range')
    device, dtype = device_and_dtype(args.precision or cfg.get('mixed_precision', 'bf16'), args.cpu)
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = MFDiT(**cfg['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    flow = build_flow(cfg)
    labels = torch.arange(cfg['model']['num_classes'], device=device).repeat(args.n_per_class)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(len(labels), cfg['model']['in_channels'],
                        cfg['model']['input_size'], cfg['model']['input_size'],
                        device=device, generator=generator)
    out = args.out or (Path(__file__).resolve().parents[1] / 'results' / 'samples') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    records = []
    with torch.inference_mode(), autocast(device, dtype):
        for steps in args.sample_steps:
            for _ in range(3):
                generate(model, flow, labels, noise, steps, args.cfg_scale)
            sync()
            durations = []
            for _ in range(args.repeats):
                sync()
                begin = time.perf_counter()
                samples = generate(model, flow, labels, noise, steps, args.cfg_scale)
                sync()
                durations.append(time.perf_counter() - begin)
            save_image(samples.float(), out / f'samples_{steps}step.png', nrow=cfg['model']['num_classes'])
            record = dict(sample_steps=steps, nfe=steps, batch_size=len(labels),
                          mean_seconds=statistics.mean(durations),
                          std_seconds=statistics.pstdev(durations),
                          milliseconds_per_image=statistics.mean(durations) * 1000 / len(labels),
                          repeats=args.repeats)
            records.append(record)
            print(record)
    write_json(out / 'timing.json', dict(checkpoint=str(args.checkpoint), training_step=checkpoint['step'],
               environment=environment(device), precision='no' if dtype is None else 'bf16',
               cfg_scale=args.cfg_scale, seed=args.seed, records=records,
               synthetic_data=cfg.get('synthetic_data', False),
               protocol='3 warmups; synchronized batch inference; excludes model loading, noise generation and PNG I/O'))
    print(f'Outputs: {out.resolve()}')


if __name__ == '__main__':
    main()
