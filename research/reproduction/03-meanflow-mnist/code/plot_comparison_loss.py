"""Plot measured (t != r) loss, as in iMF paper Figure 3.

No smoothing, clipping, fabricated values or misleading axis truncation.
Secondary diagnostic plot is optional. Runs independently of torch/CUDA.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No logged rows: {path}')
    if 'non_boundary_mse' not in rows[0]:
        raise ValueError(f'Missing non_boundary_mse (do not substitute total loss): {path}')
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mf-loss-csv', type=Path)
    p.add_argument('--imf-loss-csv', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--diagnostics-out', type=Path)
    args = p.parse_args()
    paths = {'MF': args.mf_loss_csv, 'iMF': args.imf_loss_csv}
    histories = {label: load_csv(path) for label, path in paths.items() if path}
    if not histories:
        p.error('Provide at least one training loss CSV')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for label, rows in histories.items():
        ax.plot([int(r['step']) for r in rows],
                [float(r['non_boundary_mse']) for r in rows],
                linewidth=1.1, label=label)
    ax.set(xlabel='Training steps', ylabel='Loss (t != r)',
           title='MF vs iMF: non-boundary velocity MSE (MNIST)')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print('Saved Figure-3-style raw MSE:', args.out)

    if args.diagnostics_out:
        keys = [('loss', 'All-samples mean loss'),
                ('grad_norm', 'Gradient norm (before clipping)'),
                ('tangent_rms', 'JVP direction RMS'),
                ('jvp_rms', 'JVP output RMS')]
        fig, axes = plt.subplots(len(keys), 1, figsize=(8.5, 10), sharex=True)
        for ax, (key, label) in zip(axes, keys):
            for name, rows in histories.items():
                if key not in rows[0]:
                    continue
                ax.plot([int(r['step']) for r in rows],
                        [float(r[key]) for r in rows], label=name, linewidth=1)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            ax.legend()
        axes[-1].set_xlabel('Training steps')
        fig.tight_layout()
        args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.diagnostics_out, dpi=180)
        plt.close(fig)
        print('Saved diagnostics:', args.diagnostics_out)


if __name__ == '__main__':
    main()
