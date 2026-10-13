"""Train One-shot U-Net baseline on EXACT training split/config of an existing cDDIM.

One-shot baseline learns z, user_position -> normalized complex beamspace H directly.
Unlike cDDIM, the U-Net predicts clean H (not injected noise) and samples once.
It never modifies the cDDIM checkpoint or outputs/paper.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ConditionalUNet
from dataset import ChannelDataset, load_pairs


def get_device(choice):
    device = 'cuda' if (choice == 'auto' and torch.cuda.is_available()) else ('cpu' if choice == 'auto' else choice)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; check conda/PyTorch or use --device cpu')
    if device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return device


def main():
    p = argparse.ArgumentParser(description='Train one-step U-Net from the same samples as cDDIM.')
    p.add_argument('--cddim-checkpoint', default='outputs/paper/latest.pt')
    p.add_argument('--output', default='outputs/oneshot')
    p.add_argument('--epochs', type=int, default=None, help='Default: number of epochs from cDDIM checkpoint')
    p.add_argument('--batch-size', type=int, default=None, help='Default: cDDIM batch size')
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    a = p.parse_args()
    if a.save_every < 1: p.error('--save-every must be >=1')
    cp = Path(a.cddim_checkpoint)
    if not cp.is_file(): raise FileNotFoundError(f'Existing cDDIM checkpoint missing: {cp}')
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    cfg = ck['config']
    epochs = int(a.epochs if a.epochs is not None else ck['epoch'])
    batch_size = int(a.batch_size if a.batch_size is not None else cfg['batch_size'])
    if epochs <= 0 or batch_size <= 0: p.error('epochs / batch size must be positive')
    train_start = int(cfg['train_start']); train_count = int(cfg['actual_train_samples'])
    train_end = train_start + train_count
    seed = int(cfg['seed']); n_feat = int(cfg['n_feat']); pos_scale = float(cfg['position_scale'])
    device = get_device(a.device)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda': torch.cuda.manual_seed_all(seed)
    h_all, pos_all = load_pairs(cfg['data'])
    if train_end > len(h_all): raise ValueError('Checkpoint training indices outside actual data')
    ds = ChannelDataset(h_all[train_start:train_end], pos_all[train_start:train_end], position_scale=pos_scale)
    loader = DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=(device=='cuda'))
    net = ConditionalUNet(n_feat=n_feat).to(device)
    optim = torch.optim.Adam(net.parameters(),lr=float(cfg['lr']))
    out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
    current_cfg=dict(data=str(cfg['data']),train_start=train_start,actual_train_samples=train_count,
                     batch_size=batch_size,n_feat=n_feat,position_scale=pos_scale,seed=seed,
                     epochs=epochs,learning_rate=float(cfg['lr']),constant_time_condition=1.0,
                     reference_cddim_checkpoint=str(cp),reference_epoch=int(ck['epoch']),
                     method='One-shot U-Net: z + position -> clean beamspace H; direct MSE')
    epoch0=0
    loss_path=out/'loss.csv'
    if a.resume:
        if not (out/'latest.pt').is_file(): raise FileNotFoundError('Cannot --resume: outputs/oneshot/latest.pt not found')
        resume=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if any(resume['config'].get(k)!=current_cfg[k] for k in ('data','train_start','actual_train_samples','n_feat','position_scale')):
            raise ValueError('Resume setting / training data mismatch')
        net.load_state_dict(resume['model']);optim.load_state_dict(resume['optimizer']);epoch0=int(resume['epoch'])
        if not loss_path.is_file(): raise FileNotFoundError('Cannot resume without original loss.csv')
        with loss_path.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
        if not rows or int(rows[-1]['epoch']) != epoch0:
            raise RuntimeError('Loss CSV and checkpoint epochs differ')
    elif (out/'latest.pt').exists():
        raise FileExistsError('outputs/oneshot/latest.pt already exists. Use --resume or move it first to avoid overwriting weights.')
    (out/'config.json').write_text(json.dumps(current_cfg,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'One-shot U-Net | device={device} | train={len(ds)} | epochs={epoch0}->{epochs} '
          f'| params={sum(q.numel() for q in net.parameters()):,} | data={cfg["data"]}',flush=True)
    with loss_path.open('a' if a.resume else 'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        if not a.resume: writer.writerow(['epoch','mean_train_loss','elapsed_seconds'])
        for ep in range(epoch0,epochs):
            start=time.monotonic();net.train();total=0.;seen=0
            for clean,positions in loader:
                clean=clean.to(device,non_blocking=True)
                positions=positions.to(device,non_blocking=True)
                z=torch.randn_like(clean)
                t=torch.ones(len(clean),device=device)
                pred=net(z,positions,t)
                loss=nn.functional.mse_loss(pred,clean)
                optim.zero_grad(set_to_none=True)
                loss.backward();optim.step()
                total+=float(loss.item())*len(clean);seen+=len(clean)
            mean=total/seen
            print(f'One-shot epoch {ep+1}/{epochs}  loss={mean:.6f}  seconds={time.monotonic()-start:.1f}',flush=True)
            writer.writerow([ep+1,f'{mean:.8f}',f'{time.monotonic()-start:.2f}']);f.flush()
            state=dict(model=net.state_dict(),optimizer=optim.state_dict(),epoch=ep+1,config=current_cfg)
            torch.save(state,out/'latest.pt')
            if ((ep+1)%a.save_every==0 or ep+1==epochs):
                torch.save(state,out/f'epoch_{ep+1:04d}.pt')
    print(f'One-shot training finished! checkpoint={out/"latest.pt"}',flush=True)


if __name__ == '__main__': main()
