"""Plot recorded losses from upstream train.log or train_repro.py loss.csv."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
METRICS = {'Loss': 'loss', 'FM_Loss': 'fm_loss',
           'MF_Loss': 'mf_loss', 'MF_V_MSE': 'mf_v_mse'}


def read_history(path):
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.csv':
        rows = []
        for item in csv.DictReader(text.splitlines()):
            row = {'step': int(item['step']), 'loss': float(item['loss'])}
            for key in METRICS.values():
                if item.get(key):
                    row[key] = float(item[key])
            rows.append(row)
    else:
        rows, step = [], None
        for line in text.splitlines():
            match = re.search(r'\bStep:\s*(\d+)\b', line)
            if match:
                step = int(match.group(1))
            if step is not None and re.match(r'\s*Loss:', line):
                row = {'step': step}
                for label, key in METRICS.items():
                    match = re.search(rf'\b{label}:\s*({NUMBER})(?![\w.])', line)
                    if match:
                        row[key] = float(match.group(1))
                if 'loss' not in row:
                    raise ValueError(f'Invalid Loss at step {step}')
                rows.append(row)
                step = None
    if not rows:
        raise ValueError('No recorded training losses found; supply train.log or loss.csv')
    for i, row in enumerate(rows):
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f'Non-finite training metric at step {row["step"]}')
        if i and row['step'] <= rows[i - 1]['step']:
            raise ValueError('Training steps repeat or go backwards; choose one continuous run')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = read_history(args.input)
    args.out.mkdir(parents=True, exist_ok=False)
    columns = ['step'] + [key for key in METRICS.values() if any(key in r for r in rows)]
    with (args.out / 'loss.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for key in ['loss', 'fm_loss', 'mf_loss']:
        selected = [r for r in rows if key in r]
        if selected:
            axes[0].plot([r['step'] for r in selected], [r[key] for r in selected], label=key)
    selected = [r for r in rows if 'mf_v_mse' in r]
    if selected:
        axes[1].plot([r['step'] for r in selected], [r['mf_v_mse'] for r in selected], label='mf_v_mse')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'MF velocity MSE was not recorded', ha='center', transform=axes[1].transAxes)
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Recorded loss')
    axes[1].set_ylabel('Raw velocity MSE')
    axes[1].set_xlabel('Training step')
    fig.suptitle('Recorded training metrics (no smoothing or interpolation)')
    fig.tight_layout()
    fig.savefig(args.out / 'loss_curve.png', dpi=160)
    plt.close(fig)
    metadata = dict(source=str(args.input), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    recorded_points=len(rows), first_recorded_step=rows[0]['step'],
                    last_recorded_step=rows[-1]['step'], smoothing=False,
                    note='Only recorded points; this plot does not establish unlogged training history')
    (args.out / 'loss_source.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Plotted {len(rows)} recorded points, steps {rows[0]["step"]} to {rows[-1]["step"]}: {args.out.resolve()}')


if __name__ == '__main__':
    main()
