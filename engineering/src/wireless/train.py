"""Train, evaluate on independent simulation seeds, and export browser artifacts."""
import argparse
import copy
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from model import Denoiser, simulate, schedule, sample, FS, STEPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--evaluate-only', action='store_true', help='Reuse output/model.pt and training history')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'results'/'wireless')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = args.device
    clean, observed, mask, _ = [torch.tensor(a, device=device) for a in simulate(4096, 42)]
    model = Denoiser().to(device)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    _, _, abar, _ = schedule(device)
    losses = []
    start = time.perf_counter()
    if args.evaluate_only:
        checkpoint = torch.load(args.output/'model.pt', map_location=device, weights_only=True)
        ema.load_state_dict(checkpoint['state_dict'])
        args.iterations = checkpoint['iterations']
        losses = checkpoint['losses']
    for step in range(0 if args.evaluate_only else args.iterations):
        idx = torch.randint(len(clean), (64,), device=device)
        t = torch.randint(STEPS, (64,), device=device)
        noise = torch.randn_like(clean[idx])
        a = abar[t, None, None]
        xt = a.sqrt()*clean[idx] + (1-a).sqrt()*noise
        prediction = model(xt, t, observed[idx], mask[idx])
        loss = torch.nn.functional.mse_loss(prediction, noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            for p, q in zip(ema.parameters(), model.parameters()):
                p.lerp_(q, .01)
        losses.append(float(loss.detach()))
        if (step+1) % 250 == 0:
            print(f'{step+1}/{args.iterations}: MSE={np.mean(losses[-100:]):.5f}', flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = checkpoint['training_seconds'] if args.evaluate_only else time.perf_counter()-start
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': ema.cpu().state_dict(), 'iterations': args.iterations, 'seed':42,
                'losses':losses, 'training_seconds':seconds}, args.output/'model.pt')
    ema.to(device)
    x, y, m, rate = simulate(128, 20261018)
    with torch.no_grad():
        prediction, frames = sample(ema, torch.tensor(y, device=device), torch.tensor(m, device=device), keep_frames=True)
    result = prediction.cpu().numpy()
    # Mask-aware moving average: a deliberately simple, reproducible baseline.
    def average(a):
        return np.lib.stride_tricks.sliding_window_view(np.pad(a, ((0,0),(0,0),(2,2)), mode='edge'), 5, axis=-1).mean(-1)
    smooth = average(y) / np.maximum(average(m), .01)
    def rmse(a):
        return float(np.sqrt(np.mean((a-x)**2)))
    def frequencies(a):
        spectrum = np.abs(np.fft.rfft(a-a.mean(-1, keepdims=True), axis=-1))**2
        bins = np.fft.rfftfreq(a.shape[-1], 1/FS)
        valid = (bins >= .15) & (bins <= 1.8)
        return bins[valid][np.argmax(spectrum[:,:,valid].mean(axis=1), axis=-1)]
    metrics = {'dataset':'synthetic multipath CSI amplitude; not real captured Wi-Fi',
               'train_windows':4096, 'test_windows':128, 'train_seed':42, 'test_seed':20261018,
               'sample_seed':2026, 'iterations':args.iterations, 'diffusion_steps':STEPS,
               'sample_rate_hz':FS, 'window_seconds':12.8, 'parameters':sum(p.numel() for p in ema.parameters()),
               'training_seconds':round(seconds,2), 'device':device, 'python':platform.python_version(),
               'torch':torch.__version__, 'numpy':np.__version__,
               'first_100_loss':float(np.mean(losses[:100])), 'last_100_loss':float(np.mean(losses[-100:])),
               'rmse':{'observed':rmse(y), 'moving_average':rmse(smooth), 'ddpm':rmse(result)},
               'motion_frequency_mae_hz':{name:float(np.mean(np.abs(frequencies(a)-rate))) for name,a in [('observed',y),('moving_average',smooth),('ddpm',result)]}}
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    np.savez_compressed(args.output/'evaluation.npz', clean=x, observed=y, mask=m, recovered=result, baseline=smooth,
                        frequency=rate, loss=losses, frame_steps=[t for t,_ in frames], frame_values=np.stack([v[0] for _,v in frames]))
    # First eight holdout examples, without selecting favorable outcomes.
    demo = {'metrics':metrics, 'loss':[round(float(np.mean(losses[i:i+25])),5) for i in range(0,len(losses),25)],
            'examples':[{'frequency':round(float(rate[i]),3), 'clean':x[i].round(4).tolist(),
                         'observed':y[i].round(4).tolist(), 'recovered':result[i].round(4).tolist(),
                         'baseline':smooth[i].round(4).tolist(), 'mask':m[i].astype(int).tolist(),
                         'frames':[{'t':t,'values':v[i].round(4).tolist()} for t,v in frames]}
                        for i in range(8)]}
    from export_demo import export
    export(demo)
    # Separate process avoids mixing PyTorch and plotting OpenMP runtimes on Windows.
    import subprocess
    import sys
    subprocess.run([sys.executable, str(Path(__file__).with_name('plots.py')), str(args.output)], check=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
