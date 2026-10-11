"""Checks the real JVP/backprop, resume state, history, and sampling NFE."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.dit import MFDiT
from repro_utils import build_flow, generate


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first, resumed, sampled = root / 'first', root / 'resumed', root / 'sampled'
        run('train_repro.py', '--smoke', '--cpu', '--out', first,
            '--log-every', 1, '--save-every', 1, '--sample-every', 2)
        old = first / 'ckpts' / 'step_0000002.pt'
        run('train_repro.py', '--cpu', '--resume', old, '--steps', 4,
            '--out', resumed, '--log-every', 1, '--save-every', 1, '--sample-every', 2)
        latest = resumed / 'ckpts' / 'step_0000004.pt'
        original = torch.load(old, map_location='cpu', weights_only=False)
        result = torch.load(latest, map_location='cpu', weights_only=False)
        assert original['step'] == 2 and result['step'] == 4
        assert result['history'][:len(original['history'])] == original['history']
        assert result['config']['synthetic_data'] is True
        assert any(not torch.equal(value, result['model'][key]) for key, value in original['model'].items())
        assert result['optimizer']['state']
        assert result['scheduler']['last_epoch'] == 4
        with (resumed / 'loss.csv').open() as f:
            assert len(list(csv.DictReader(f))) == 4
        assert (resumed / 'loss_curve.png').is_file()
        run('sample_repro.py', '--checkpoint', latest, '--cpu', '--n-per-class', 1,
            '--repeats', 2, '--out', sampled)
        records = json.loads((sampled / 'timing.json').read_text())['records']
        assert [row['nfe'] for row in records] == [1, 5]
        model = MFDiT(**result['config']['model'])
        model.load_state_dict(result['model'])
        model.eval()
        calls = []
        hook = model.register_forward_hook(lambda *args: calls.append(1))
        labels = torch.arange(10)
        noise = torch.randn(10, 1, 32, 32)
        for steps in [1, 5]:
            calls.clear()
            images = generate(model, build_flow(result['config']), labels, noise, n_steps=steps)
            assert len(calls) == steps
            assert images.shape == noise.shape and torch.isfinite(images).all()
            assert images.min() >= 0 and images.max() <= 1
        hook.remove()
        print('PASS: JVP/backprop, checkpoint/resume, real history, 1/5 NFE, finite image output')


if __name__ == '__main__':
    main()
