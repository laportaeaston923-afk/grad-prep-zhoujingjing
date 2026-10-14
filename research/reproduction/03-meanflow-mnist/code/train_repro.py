"""Single GPU runner using the pinned upstream loss and DiT."""
import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from data import build_dataset, cycle
from models.dit import MFDiT
from repro_utils import (autocast, build_flow, device_and_dtype, environment,
                         generate, load_config, plot_history, write_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/mnist_laptop.py')
    p.add_argument('--steps', type=int, help='Total target step, including resumed steps')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--precision', choices=['bf16', 'no'])
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Tiny synthetic test, not an experiment')
    p.add_argument('--resume', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--sample-every', type=int, default=500)
    args = p.parse_args()
    if min(args.log_every, args.save_every, args.sample_every) < 1:
        p.error('Intervals must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    cfg = checkpoint['config'] if checkpoint else load_config(args.config)
    cfg['synthetic_data'] = args.smoke or cfg.get('synthetic_data', False)
    if cfg.get('latent', False):
        p.error('This runner supports pixel MNIST/CIFAR-10 only; use upstream train.py for latent data')
    if args.smoke and checkpoint is None:
        cfg['model'].update(dim=32, depth=1, num_heads=4, patch_size=4)
        cfg.update(n_steps=2, batch_size=2, num_workers=0)
    if args.steps is not None:
        cfg['n_steps'] = args.steps
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.precision is not None:
        cfg['mixed_precision'] = args.precision
    if cfg['n_steps'] < 1 or cfg['batch_size'] < 1:
        p.error('Steps and batch size must be positive')
    device, dtype = device_and_dtype(cfg.get('mixed_precision', 'bf16'), args.cpu)
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = MFDiT(**cfg['model']).to(device)
    flow = build_flow(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = cfg.get('scheduler', {})
    schedule_steps = checkpoint.get('schedule_steps', cfg['n_steps']) if checkpoint else cfg['n_steps']
    warmup = sched.get('warmup_steps', 0)
    minimum = sched.get('min_lr_ratio', 0.1)

    def lr_scale(step):
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        fraction = min(1.0, max(0.0, (step - warmup) / max(schedule_steps - warmup, 1)))
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale) if sched else None
    start = 0
    history = []
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and checkpoint['scheduler']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start = checkpoint['step']
        history = checkpoint['history']
    if cfg['n_steps'] <= start:
        p.error('Target --steps must exceed the checkpoint step')
    out = args.out or (Path(__file__).resolve().parents[1] / 'results' / 'runs') / f"{cfg['name']}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    (out / 'ckpts').mkdir()
    write_json(out / 'config.json', cfg)
    metadata = environment(device)
    metadata.update(smoke=cfg['synthetic_data'], resumed_from=str(args.resume) if args.resume else None,
                    parameters=sum(x.numel() for x in model.parameters()),
                    schedule_steps=schedule_steps, seed=args.seed,
                    precision='no' if dtype is None else 'bf16')
    write_json(out / 'environment.json', metadata)
    if cfg['synthetic_data']:
        from torchvision.datasets import FakeData
        from torchvision.transforms import ToTensor
        dataset = FakeData(size=16, image_size=(cfg['model']['in_channels'], 32, 32),
                           num_classes=10, transform=ToTensor())
    else:
        dataset = build_dataset(cfg)
    if len(dataset) < cfg['batch_size']:
        p.error('Batch size exceeds dataset size')
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                                        num_workers=cfg['num_workers'], drop_last=True,
                                        pin_memory=device.type == 'cuda')
    batches = cycle(loader)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    labels = torch.arange(cfg['model']['num_classes'], device=device).repeat(10)
    fixed_noise = torch.randn(len(labels), cfg['model']['in_channels'],
                              cfg['model']['input_size'], cfg['model']['input_size'],
                              device=device, generator=generator)
    if checkpoint:
        random.setstate(checkpoint['rng_python'])
        np.random.set_state(checkpoint['rng_numpy'])
        torch.set_rng_state(checkpoint['rng_torch'])
        if device.type == 'cuda' and checkpoint['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    # Controlled MF/iMF uses the u-boundary condition, not the legacy v-head.
    # Do not invent an auxiliary FM loss for runs that do not train such a head.
    metric_names = (['mf_loss', 'mf_v_mse', 'non_boundary_mse'] if 'comparison' in cfg
                    else ['fm_loss', 'mf_loss', 'mf_v_mse'])
    columns = ['step', 'interval_steps', 'loss', *metric_names, 'lr', 'grad_norm']
    values = {k: 0.0 for k in ['loss', *metric_names, 'grad_norm']}
    count = 0
    last_saved = None

    def save(step):
        nonlocal last_saved
        target = out / 'ckpts' / f'step_{step:07d}.pt'
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict() if scheduler else None,
                       config=cfg, step=step, history=history, schedule_steps=schedule_steps,
                       rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                       rng_torch=torch.get_rng_state(),
                       rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
        temporary = target.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(target)
        last_saved = target

    print(f"Device: {device}; parameters: {metadata['parameters'] / 1e6:.2f}M; mode: {flow.mode}")
    print(f'Outputs: {out.resolve()}')
    with (out / 'loss.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
        f.flush()
        save(start)
        try:
            progress = tqdm(range(start + 1, cfg['n_steps'] + 1), desc=cfg['name'])
            for step in progress:
                model.train()
                x, c = next(batches)
                x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, dtype):
                    loss, metrics = flow.loss(model, x, c)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; retry FP32 with --precision no')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                values['loss'] += loss.item()
                values['grad_norm'] += grad_norm.item()
                for key in metrics:
                    values[key] += metrics[key].item()
                count += 1
                progress.set_postfix(loss=f'{loss.item():.4f}', mse=f"{metrics['mf_v_mse'].item():.4f}")
                save_now = step % args.save_every == 0 or step == cfg['n_steps']
                if step % args.log_every == 0 or save_now:
                    row = dict(step=step, interval_steps=count, lr=optimizer.param_groups[0]['lr'],
                               **{k: v / count for k, v in values.items()})
                    history.append(row)
                    writer.writerow(row)
                    f.flush()
                    values = {k: 0.0 for k in values}
                    count = 0
                if save_now:
                    save(step)
                if step % args.sample_every == 0 or step == cfg['n_steps']:
                    model.eval()
                    with autocast(device, dtype):
                        samples = generate(model, flow, labels, fixed_noise, n_steps=1, cfg_scale=1.0)
                    save_image(samples.float(), out / 'images' / f'step_{step:07d}_1step.png', nrow=10)
                    plot_history(history, out / 'loss_curve.png')
        except KeyboardInterrupt:
            print(f'Interrupted. Resume the last complete checkpoint: {last_saved}')
            print('Updates after that checkpoint are not restored. Original outputs are kept.')
            plot_history(history, out / 'loss_curve.png')
            return
    print(f'Training finished: {last_saved}')


if __name__ == '__main__':
    main()
