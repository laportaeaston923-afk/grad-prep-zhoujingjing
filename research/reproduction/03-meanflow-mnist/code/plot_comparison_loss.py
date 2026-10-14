"""Plot the MF/iMF training CSV in a fresh process without importing PyTorch.

This avoids late matplotlib + libiomp5md collisions inside long-running Windows
training processes. Pass either or both CSV inputs.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_history(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No training logs in {path}')
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mf-loss-csv', type=Path)
    p.add_argument('--imf-loss-csv', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not (args.mf_loss_csv or args.imf_loss_csv):
        p.error('Provide at least one CSV')
    histories = {}
    if args.mf_loss_csv:
        histories['MF'] = read_history(args.mf_loss_csv)
    if args.imf_loss_csv:
        histories['iMF'] = read_history(args.imf_loss_csv)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, rows in histories.items():
        steps = [int(row['step']) for row in rows]
        axes[0].plot(steps, [float(row['loss']) for row in rows], label=name)
        # Paper Fig. 3 focuses on non-boundary training errors.
        key = ('non_boundary_mse' if 'non_boundary_mse' in rows[0]
               else 'mf_v_mse')
        axes[1].plot(steps, [float(row[key]) for row in rows], label=name)
    axes[0].set_ylabel('Training loss (interval mean)')
    axes[1].set_ylabel('Non-boundary raw MSE')
    axes[1].set_xlabel('Training step')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print(f'Loss curve saved: {args.out}')


if __name__ == '__main__':
    main()
