"""Run matched MF/iMF MNIST training and ONE-step evaluation only.

Run from code/: python run_mf_imf_comparison.py
The script never fabricates results: summary is written only after both
subprocesses finish. Existing results are never overwritten.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(*args):
    print('\n$', ' '.join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--fid-batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--precision', choices=['bf16', 'no'])
    parser.add_argument('--skip-fid', action='store_true')
    parser.add_argument('--skip-timing', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.num_samples < 2 or min(args.batch_size, args.fid_batch_size) < 1:
        parser.error('Steps and batches must be positive; FID requires >=2 samples')
    precision = args.precision or ('no' if args.cpu else 'bf16')
    root = Path(__file__).resolve().parent
    out = args.out or (root.parent / 'results' / 'mf-vs-imf-one-step' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    device_flags = ['--cpu'] if args.cpu else []
    records = {}
    for variant in ('mf', 'imf'):
        run_dir = out / variant
        checkpoint_dir = run_dir / 'train' / 'ckpts'
        config = root / 'configs' / f'mnist_compare_{variant}.py'
        run(sys.executable, root / 'train_repro.py', '--config', config,
            '--steps', args.steps, '--batch-size', args.batch_size,
            '--precision', precision, '--seed', args.seed,
            '--log-every', min(100, args.steps),
            '--save-every', args.steps, '--sample-every', args.steps,
            '--out', run_dir / 'train', *device_flags)
        checkpoint = checkpoint_dir / f'step_{args.steps:07d}.pt'
        if not checkpoint.exists():
            raise RuntimeError(f'Expected trained checkpoint missing: {checkpoint}')
        row = {'variant': variant, 'checkpoint': str(checkpoint), 'steps': args.steps}
        with (run_dir / 'train' / 'loss.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        row['final_interval_loss'] = float(rows[-1]['loss'])
        if not args.skip_fid:
            run(sys.executable, root / 'eval_fid.py', '--checkpoint', checkpoint,
                '--num-samples', args.num_samples, '--sample-steps', 1,
                '--batch-size', args.batch_size,
                '--fid-batch-size', args.fid_batch_size,
                '--seed', args.seed, '--cfg-scale', 1,
                '--precision', 'no', '--out', run_dir / 'fid', *device_flags)
            with (run_dir / 'fid' / 'fid.json').open(encoding='utf-8') as f:
                row['fid_1step'] = json.load(f)['results'][0]['fid']
        if not args.skip_timing:
            run(sys.executable, root / 'sample_repro.py', '--checkpoint', checkpoint,
                '--sample-steps', 1, '--n-per-class', 10,
                '--cfg-scale', 1, '--precision', precision,
                '--seed', args.seed, '--out', run_dir / 'timing', *device_flags)
            with (run_dir / 'timing' / 'timing.json').open(encoding='utf-8') as f:
                row['milliseconds_per_image'] = json.load(f)['records'][0]['milliseconds_per_image']
        records[variant] = row
        (out / 'partial_results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')

    result = {'kind': 'controlled_mf_vs_imf_one_step',
              'note': 'No CFG, same seed/data/model/loss. Boundary u(t,t) vs sample e-x JVP tangent.',
              'precision_training': precision, 'fid_precision': 'FP32',
              'n_samples_per_setting': args.num_samples, 'records': records}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    header = '| 方法 | 训练末段 Loss | 一步 FID↓ | 一步耗时 ms/图↓ |\n|---|---:|---:|---:|\n'
    lines = []
    for name in ('mf', 'imf'):
        r = records[name]
        fid = f"{r['fid_1step']:.4f}" if 'fid_1step' in r else '未评估'
        speed = f"{r['milliseconds_per_image']:.3f}" if 'milliseconds_per_image' in r else '未测量'
        lines.append(f"| {name.upper()} | {r['final_interval_loss']:.4f} | {fid} | {speed} |")
    (out / 'README.md').write_text(
        '# MF vs iMF · MNIST 一步生成对比\n\n'
        '两组分别从头训练；对照的唯一算法差异为 JVP 方向。'
        '训练不使用 CFG、辅助瞬时速度头或五步采样。'
        '末段 Loss 为区间均值，不代替 FID。\n\n'
        + header + '\n'.join(lines) + '\n\n'
        '详细配置、训练 Loss 和权重在 `mf/train/`、`imf/train/`；'
        'FID 在各自 `fid/`，计时在 `timing/`。'
        '若样本数小于 10,000，FID 仅为初步检查，不应作为正式结果。\n',
        encoding='utf-8')
    print(f'Comparison complete: {out / "README.md"}')


if __name__ == '__main__':
    main()
