import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from model import RepairNet,schedule,make_masks,T

ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=8000);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.iterations<1:p.error('iterations must be positive')
    torch.set_num_threads(4);torch.manual_seed(42);rng=np.random.default_rng(42);device=args.device
    data=np.load(ROOT/'data'/'mnist.npz');x=torch.tensor(data['train'][:,None],device=device,dtype=torch.float32)/127.5-1
    print('Preparing a reusable pool of 4096 random masks...',flush=True)
    mask_pool=torch.tensor(make_masks(4096,rng=rng),device=device)
    if device.startswith('cuda'):torch.backends.cudnn.benchmark=True
    model=RepairNet().to(device);old_steps=0
    if args.resume:
        saved=torch.load(ROOT/'results'/'repair.pt',map_location=device,weights_only=True);model.load_state_dict(saved['weights']);old_steps=saved['training']['iterations']
    ema=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=3e-4);_,abar=schedule(device);losses=[];start=time.perf_counter()
    for step in range(args.iterations):
        idx=torch.randint(len(x),(128,),device=device);clean=x[idx];hole=mask_pool[torch.randint(len(mask_pool),(len(idx),),device=device)];observed=clean*(1-hole)
        t=torch.randint(T,(len(idx),),device=device);a=abar[t,None,None,None];noise=torch.randn_like(clean);xt=a.sqrt()*clean+(1-a).sqrt()*noise
        with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
            pred=model(xt,t,observed,hole)
        weight=hole+.1*(1-hole);loss=((pred.float()-noise).square()*weight).sum()/weight.sum()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        with torch.no_grad():
            for e,q in zip(ema.parameters(),model.parameters()):e.lerp_(q,.005)
        losses.append(float(loss.detach()))
        if (step+1)%500==0:print(f'{step+1}/{args.iterations}: noise MSE {np.mean(losses[-100:]):.5f}',flush=True)
        if (step+1)%2000==0:
            partial={'iterations':step+1+old_steps,'seed':42,'mask_pool':4096}
            torch.save({'weights':{k:v.cpu() for k,v in ema.state_dict().items()},'training':partial},ROOT/'results'/'repair.pt')
    if device.startswith('cuda'):torch.cuda.synchronize()
    info={'iterations':args.iterations+old_steps,'current_run_iterations':args.iterations,'seconds':round(time.perf_counter()-start,2),'seed':42,'batch_size':128,'learning_rate':.0003,'parameters':sum(p.numel() for p in model.parameters()),'device':device,'hardware':torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU','torch':str(torch.__version__),'train_samples':len(x),'diffusion_steps':100,'mask_pool':4096,'mixed_precision':'bfloat16' if device.startswith('cuda') else 'float32'}
    torch.save({'weights':ema.cpu().state_dict(),'training':info},ROOT/'results'/'repair.pt')
    (ROOT/'results'/'training.json').write_text(json.dumps({**info,'loss':[round(float(np.mean(losses[i:i+25])),6) for i in range(0,len(losses),25)]},indent=2),encoding='utf-8')
    print(info,flush=True)

