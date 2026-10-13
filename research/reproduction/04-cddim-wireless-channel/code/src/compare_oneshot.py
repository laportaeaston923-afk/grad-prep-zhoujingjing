"""Compare saved cDDIM ID/OOD CDF with One-shot U-Net on identical test positions.

Does NOT regenerate cDDIM curves, retrain cDDIM, or overwrite existing plots.
This is a two-model Fig 6/7-style supplement, not the full paper protocol.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ConditionalUNet
from dataset import load_pairs, to_model_array


def peak(array):
    """Same Tx peak rule as Fig. 6 and 7: max over Rx, argmax over Tx."""
    amp = np.hypot(array[:, 0], array[:, 1])
    return amp.max(axis=1).argmax(axis=1).astype(np.int64)


def device_for(name):
    dev = 'cuda' if name == 'auto' and torch.cuda.is_available() else ('cpu' if name == 'auto' else name)
    if dev == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable')
    if dev == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return dev


def load_saved_cdf(path, expected_count, expected_summary, want_seed=42):
    if not path.is_file():
        raise FileNotFoundError(f'Missing existing cDDIM CDF: {path}; run Fig 6/7 first')
    if not expected_summary.is_file():
        raise FileNotFoundError(f'Missing existing cDDIM summary: {expected_summary}')
    summary = json.loads(expected_summary.read_text(encoding='utf-8'))
    count = int(summary.get('number_of_test_channels', summary.get('ood_count', -1)))
    if count != expected_count or int(summary['seed']) != want_seed:
        raise ValueError('Existing cDDIM CDF evaluation count/seed differs from One-shot')
    data = np.loadtxt(path, delimiter=',', skiprows=1, encoding='utf-8-sig')
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f'Unexpected CDF CSV shape in {path}')
    return data[:, 0].astype(int), data[:, 1], summary


def generate(net, positions, scale, device, seed, batch_size):
    preds = []
    for offset in range(0, len(positions), batch_size):
        stop = min(offset + batch_size, len(positions))
        pos = torch.from_numpy((positions[offset:stop] / scale).astype('float32')).to(device)
        g = torch.Generator(device=device).manual_seed(seed + offset)
        z = torch.randn((len(pos), 2, 4, 32), generator=g, device=device)
        t = torch.ones(len(pos), device=device)
        with torch.inference_mode():
            sample = net(z, pos, t).cpu().numpy()
        preds.append(sample)
        print(f'  One-shot generated {stop}/{len(positions)}', flush=True)
    return np.concatenate(preds, axis=0)


def save_rows(path, truth, gen, distances=None):
    err = np.abs(truth - gen)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_offset','reference_tx_peak','oneshot_tx_peak','absolute_error','radius_m'])
        for i in range(len(err)):
            writer.writerow([i, int(truth[i]), int(gen[i]), int(err[i]),
                             '' if distances is None else float(distances[i])])
    return err


def draw(d, cddim, oneshot, caption, output):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout='constrained')
    ax.plot(d, cddim, 'o-', label='cDDIM (saved result)', linewidth=2, markersize=4)
    ax.plot(d, oneshot, 's-', label='One-shot U-Net (new baseline)', linewidth=2, markersize=4)
    ax.set(xlabel='Absolute Tx peak-index difference D', ylabel='P(error <= D)',
           title=caption, ylim=(0, 1.02), xlim=(0, int(d.max())))
    ax.set_xticks(d)
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output, dpi=175)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Evaluate One-shot and compare to saved cDDIM Fig6/7')
    p.add_argument('--oneshot-checkpoint', default='outputs/oneshot/latest.pt')
    p.add_argument('--cddim-checkpoint', default='outputs/paper/latest.pt')
    p.add_argument('--ood-data', default='data/QuaDRiGa/NumUEs_500_BerUMaL_ULA_OOD_100to200m.mat')
    p.add_argument('--output', default='outputs/oneshot/comparison')
    p.add_argument('--id-start', type=int, default=90000)
    p.add_argument('--count', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    a = p.parse_args()
    if a.count < 1 or a.batch_size < 1:
        p.error('count/batch size must be positive')
    o_ck = torch.load(a.oneshot_checkpoint, map_location='cpu', weights_only=False)
    c_ck = torch.load(a.cddim_checkpoint, map_location='cpu', weights_only=False)
    cfg = o_ck['config']
    ref_cfg = c_ck['config']
    for key in ('data','train_start','actual_train_samples','n_feat','position_scale'):
        if str(cfg[key]) != str(ref_cfg[key]):
            raise ValueError(f'Mismatch between cDDIM and One-shot config: {key}')
    if o_ck['epoch'] != c_ck['epoch']:
        raise ValueError('Epoch mismatch cDDIM and One-shot')
    d_id, c_id, s_id = load_saved_cdf(Path('outputs/paper/fig6_id/fig6_cdf.csv'), a.count,
                                      Path('outputs/paper/fig6_id/fig6_summary.json'), a.seed)
    d_ood, c_ood, s_ood = load_saved_cdf(Path('outputs/paper/fig7_ood/fig7_ood_cdf.csv'), a.count,
                                         Path('outputs/paper/fig7_ood/fig7_summary.json'), a.seed)
    if not np.array_equal(d_id, d_ood):
        raise ValueError('ID/OOD CDF threshold ranges differ')
    if s_id['testing_range'] != [a.id_start, a.id_start + a.count]:
        raise ValueError('ID test slice differs from saved cDDIM Fig6 slice')
    if int(s_id.get('checkpoint_epoch', -1)) != int(c_ck['epoch']) or int(s_ood.get('checkpoint_epoch', -1)) != int(c_ck['epoch']):
        raise ValueError('Saved cDDIM results were generated with a different epoch')
    if int(s_id.get('sampling_steps', -1)) != 256 or int(s_ood.get('ddim_steps', -1)) != 256:
        raise ValueError('Saved cDDIM curves were not generated with 256 steps')
    if Path(s_ood['data_source']).name != Path(a.ood_data).name:
        raise ValueError('OOD reference file mismatch')
    dev = device_for(a.device)
    net = ConditionalUNet(n_feat=int(cfg['n_feat'])).to(dev)
    net.load_state_dict(o_ck['model'])
    net.eval()
    all_h, all_pos = load_pairs(cfg['data'])
    tr_start = int(cfg['train_start'])
    tr_end = tr_start + int(cfg['actual_train_samples'])
    if a.id_start < tr_end and tr_start < a.id_start + a.count:
        raise ValueError('ID test overlaps training data')
    if a.id_start < 0 or a.id_start + a.count > len(all_h):
        raise ValueError('ID test slice outside dataset')
    x_id = all_pos[a.id_start:a.id_start+a.count]
    true_id = peak(to_model_array(all_h[a.id_start:a.id_start+a.count]))
    ood_h, x_ood = load_pairs(a.ood_data)
    if a.count > len(ood_h):
        raise ValueError('OOD file too small')
    x_ood = x_ood[:a.count]
    r_ood = np.hypot(x_ood[:, 0], x_ood[:, 1])
    if (r_ood < 100-1e-3).any() or (r_ood > 200+1e-3).any():
        raise ValueError('OOD test positions not in 100-200m annulus')
    tr_pos = all_pos[tr_start:tr_end]
    if (np.hypot(tr_pos[:, 0], tr_pos[:, 1]) > 100+1e-3).any():
        raise ValueError('Training positions outside 100m circle')
    true_ood = peak(to_model_array(ood_h[:a.count]))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Compare cDDIM epoch {c_ck["epoch"]} / One-shot epoch {o_ck["epoch"]}; N={a.count}; device={dev}', flush=True)
    pred_id = generate(net, x_id, float(cfg['position_scale']), dev, a.seed, a.batch_size)
    pred_ood = generate(net, x_ood, float(cfg['position_scale']), dev, a.seed, a.batch_size)
    err_id = save_rows(out/'oneshot_id_details.csv', true_id, peak(pred_id),
                       np.hypot(x_id[:, 0], x_id[:, 1]))
    err_ood = save_rows(out/'oneshot_ood_details.csv', true_ood, peak(pred_ood), r_ood)
    oneshot_id = np.array([(err_id <= d).mean() for d in d_id])
    oneshot_ood = np.array([(err_ood <= d).mean() for d in d_id])
    draw(d_id, c_id, oneshot_id, f'ID 0–100m | {a.count} matched test positions',
         out/'fig6_id_cddim_vs_oneshot.png')
    draw(d_id, c_ood, oneshot_ood, f'OOD 100–200m | {a.count} matched test positions',
         out/'fig7_ood_cddim_vs_oneshot.png')
    with (out/'cdf_comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['D','cddim_ID','oneshot_ID','cddim_OOD','oneshot_OOD'])
        for i, d in enumerate(d_id):
            writer.writerow([int(d), float(c_id[i]), float(oneshot_id[i]),
                             float(c_ood[i]), float(oneshot_ood[i])])
    # Optional three-way Fig5-style comparison on the identical first ID positions
    old = Path('outputs/paper/generated_vs_reference.npz')
    if old.exists():
        with np.load(old) as pair:
            old_pos = pair['positions']
            old_pred = pair['predicted']
            old_ref = pair['reference']
        n = min(5, len(old_pos))
        if n > 0 and np.allclose(old_pos[:n], x_id[:n], atol=1e-4) and old_pred.shape[1:] == (2,4,32):
            fig, axes = plt.subplots(n, 3, figsize=(9, 2*n+.6), squeeze=False, layout='constrained')
            for k in range(n):
                for j, (m, title) in enumerate(((old_pred, 'cDDIM'), (pred_id, 'One-shot U-Net'), (old_ref, 'Reference'))):
                    amplitude = np.hypot(m[k, 0], m[k, 1])
                    amplitude /= max(1e-8, float(amplitude.max()))
                    axes[k, j].imshow(amplitude, cmap='viridis', vmin=0, vmax=1,
                                      interpolation='nearest', aspect='auto')
                    if k == 0:
                        axes[k, j].set_title(title)
                    if j == 0:
                        axes[k, j].set_ylabel(f'UE {k+1} / Rx')
                    axes[k, j].set_xlabel('BS / Tx beam index')
            fig.suptitle('Supplementary Fig5-style: cDDIM / One-shot / reference (ID)', fontsize=11)
            fig.savefig(out/'fig5_three_way_ID.png', dpi=170)
            plt.close(fig)
        else:
            print('Existing Fig5 positions differ; skipping optional three-column preview')
    result = dict(scope='One-shot baseline vs saved cDDIM (not full published Fig6/7)',
                  train_n=int(cfg['actual_train_samples']), train_epochs=int(o_ck['epoch']),
                  eval_n=int(a.count), one_shot_calls_per_sample=1, cddim_calls_per_sample=256,
                  id_D0=dict(cddim=float(c_id[0]), oneshot=float(oneshot_id[0])),
                  id_D2=dict(cddim=float(c_id[2]), oneshot=float(oneshot_id[2])),
                  ood_D0=dict(cddim=float(c_ood[0]), oneshot=float(oneshot_ood[0])),
                  ood_D2=dict(cddim=float(c_ood[2]), oneshot=float(oneshot_ood[2])),
                  note='Compared at same reference positions; channel generation has random initial noise.')
    (out/'comparison_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f'Finished. New images in {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
