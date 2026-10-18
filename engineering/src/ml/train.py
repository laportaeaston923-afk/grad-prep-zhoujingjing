"""Train six networks, evaluate seven configurations, export offline Web data.

DDPM and DDIM share one checkpoint. LDM reuses the conditional VAE decoder.
MeanFlow uses the average-velocity identity with stop-gradient JVP targets.
"""
import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models import MLP,VAE,TinyDiT,schedule,diffusion,flow_sample,T

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=4500)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args()
    if args.iterations<100:parser.error('Use at least 100 iterations')
    torch.set_num_threads(4);torch.manual_seed(42)
    device=args.device;out=ROOT/'results';out.mkdir(exist_ok=True)
    d=np.load(out/'dataset.npz')
    x=torch.tensor(d['train'],device=device);y=torch.tensor(d['train_labels'],device=device)
    test=torch.tensor(d['test'],device=device);test_y=torch.tensor(d['test_labels'],device=device)
    models={'vae':VAE().to(device),'ddpm':MLP().to(device),'ldm':MLP(dim=12).to(device),
            'dit':TinyDiT().to(device),'fm':MLP().to(device),'meanflow':MLP(meanflow=True).to(device)}
    beta,abar=schedule(device);logs={};training={}
    for index,(name,model) in enumerate(models.items()):
        path=out/f'{name}.pt'
        if args.evaluate_only:
            saved=torch.load(path,map_location=device,weights_only=True)
            model.load_state_dict(saved['weights']);logs[name]=saved['loss'];training[name]=saved['training']
            continue
        torch.manual_seed(42+index)
        ema=copy.deepcopy(model);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        history=[];start=time.perf_counter()
        if name=='ldm':
            with torch.no_grad():
                mu,_=models['vae'].encode(x,y)
                latent_mean=mu.mean(0);latent_std=mu.std(0).clamp_min(.1)
                latent_data=(mu-latent_mean)/latent_std
        for step in range(args.iterations):
            idx=torch.randint(len(x),(128,),device=device);data=x[idx];labels=y[idx]
            if name=='vae':
                recon,mu,lv=model(data,labels)
                kl=.5*(mu.square()+lv.exp()-1-lv).mean()
                loss=(recon-data).square().mean()+.02*min(1,(step+1)/1000)*kl
            elif name in ['ddpm','ldm','dit']:
                if name=='ldm':data=latent_data[idx]
                t=torch.randint(T,(len(idx),),device=device);noise=torch.randn_like(data)
                a=abar[t,None];xt=a.sqrt()*data+(1-a).sqrt()*noise
                loss=(model(xt,t/T,labels)-noise).square().mean()
            elif name=='fm':
                t=torch.rand(len(idx),device=device);noise=torch.randn_like(data)
                z=(1-t[:,None])*data+t[:,None]*noise
                loss=(model(z,t,labels)-(noise-data)).square().mean()
            else:
                # Time runs from data (0) to noise (1); sampling runs backwards.
                pair=torch.sigmoid(torch.randn(len(idx),2,device=device)-.4)
                t=pair.max(-1).values;r=pair.min(-1).values
                r=torch.where(torch.rand_like(t)<.5,t,r)
                noise=torch.randn_like(data);v=noise-data
                z=(1-t[:,None])*data+t[:,None]*noise
                u,du=torch.func.jvp(lambda z,t,r:model(z,t,labels,r),
                                   (z,t,r),(v,torch.ones_like(t),torch.zeros_like(r)))
                target=(v-(t-r)[:,None]*du).detach()
                per_sample=(u-target).square().mean(-1)
                loss=(per_sample/(per_sample.detach()+.001).pow(.75)).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            with torch.no_grad():
                for p,q in zip(ema.parameters(),model.parameters()):p.lerp_(q,.01)
            history.append(float(loss.detach()))
            if (step+1)%1000==0:print(f'{name} {step+1}/{args.iterations}: {np.mean(history[-100:]):.5f}',flush=True)
        if device.startswith('cuda'):torch.cuda.synchronize()
        info={'seconds':round(time.perf_counter()-start,3),'parameters':sum(p.numel() for p in model.parameters()),'iterations':args.iterations}
        model.load_state_dict(ema.state_dict());models[name]=model.eval()
        logs[name]=[round(float(np.mean(history[i:i+25])),6) for i in range(0,len(history),25)]
        saved={'weights':model.cpu().state_dict(),'loss':logs[name],'training':info}
        if name=='ldm':saved.update(latent_mean=latent_mean.cpu(),latent_std=latent_std.cpu())
        torch.save(saved,path);model.to(device)
        training[name]=info;print(name,info,flush=True)
    # The VAE and LDM use the same frozen encoder/decoder. Latent normalization
    # is recorded in the LDM checkpoint, not computed from held-out samples.
    saved=torch.load(out/'ldm.pt',map_location=device,weights_only=True)
    lm,ls=saved['latent_mean'],saved['latent_std']
    torch.manual_seed(1042)
    classifier=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,10)).to(device)
    opt=torch.optim.AdamW(classifier.parameters(),lr=.001)
    for step in range(1200):
        idx=torch.randint(len(x),(128,),device=device)
        loss=nn.functional.cross_entropy(classifier(x[idx]),y[idx]);opt.zero_grad();loss.backward();opt.step()
    classifier.eval()
    with torch.no_grad():holdout_acc=float((classifier(test).argmax(-1)==test_y).float().mean())
    torch.save(classifier.cpu().state_dict(),out/'classifier.pt');classifier.to(device)
    def sync():
        if device.startswith('cuda'):torch.cuda.synchronize()
    @torch.no_grad()
    def generate(name,steps,seed,count=200,capture=False):
        torch.manual_seed(seed)
        labels=torch.arange(count,device=device)%10
        noise=torch.randn(count,12 if name in ['vae','ldm'] else 64,device=device)
        model=models['ddpm' if name=='ddim' else name]
        if name=='vae':
            decoded=model.decode(noise,labels)
            return decoded,[decoded.cpu()] if capture else [],labels
        if name in ['ddpm','ddim','ldm','dit']:
            values,frames=diffusion(model,labels,noise,steps,stochastic=name=='ddpm',latent=name=='ldm',keep_frames=capture)
            if name=='ldm':
                values=models['vae'].decode(values*ls+lm,labels)
                frames=[models['vae'].decode(f.to(device)*ls+lm,labels).cpu() for f in frames]
        else:values,frames=flow_sample(model,labels,noise,steps,meanflow=name=='meanflow',keep_frames=capture)
        return values.clamp(-1,1),frames,labels
    @torch.no_grad()
    def metrics(samples,labels):
        # Biased RBF MMD^2 on pixel vectors, bandwidth sigma=0.5; not FID.
        a=(samples+1)/2;b=(test+1)/2
        def kernel(u,v):return torch.exp(-torch.cdist(u,v).square()/64/(2*.5**2))
        mmd=float(kernel(a,a).mean()+kernel(b,b).mean()-2*kernel(a,b).mean())
        acc=float((classifier(samples).argmax(-1)==labels).float().mean())
        nearest=float((torch.cdist(samples,x).min(-1).values/8).mean())
        return {'mmd2':mmd,'label_consistency':acc,'nearest_train_rmse':nearest}
    configs={'ddpm':[100],'ddim':[5,10,20,50,100],'vae':[1],'ldm':[10,20,50],
             'dit':[10,20,50,100],'fm':[1,5,10,20,50],'meanflow':[1,2,4,8]}
    records=[];gallery={}
    for name,step_list in configs.items():
        for steps in step_list:
            # Warm-up excluded; timing excludes trajectory export and metric computation.
            generate(name,steps,999,count=20);scores=[];elapsed=[]
            for seed in [17,42,123]:
                sync();start=time.perf_counter();samples,frames,labels=generate(name,steps,seed);sync()
                elapsed.append((time.perf_counter()-start)*1000);scores.append(metrics(samples,labels))
                if seed==42:
                    samples,frames,labels=generate(name,steps,seed,capture=True)
                    def pixels(a):return (((a.clamp(-1,1)+1)*127.5).round().byte().numpy()).tolist()
                    # First 100 samples: exactly ten per conditioning class, no curation.
                    ids=np.unique(np.linspace(0,len(frames)-1,min(13,len(frames))).round().astype(int))
                    gallery[f'{name}-{steps}']={'samples':pixels(samples[:100].cpu()),
                        'labels':labels[:100].cpu().tolist(),'frames':[pixels(frames[i][:10]) for i in ids],
                        'frame_progress':[round(float(i/max(1,len(frames)-1)),4) for i in ids]}
            rec={'id':f'{name}-{steps}','model':name,'nfe':steps,
                 'decoder_calls':1 if name in ['vae','ldm'] else 0,
                 'batch_ms':float(np.mean(elapsed)),'batch_ms_std':float(np.std(elapsed,ddof=1)),
                 'samples':200,'seeds':[17,42,123]}
            for key in scores[0]:
                rec[key]=float(np.mean([s[key] for s in scores]));rec[key+'_std']=float(np.std([s[key] for s in scores],ddof=1))
            records.append(rec);print('evaluate',rec,flush=True)
    # VAE reconstruction is separately reported; it is not unpaired generation.
    with torch.no_grad():
        mu,_=models['vae'].encode(test,test_y);recon=models['vae'].decode(mu,test_y)
        recon_rmse=float((recon-test).square().mean().sqrt())
        test_mu,_=models['vae'].encode(test,test_y)
        # A class-0 latent interpolation between two real test examples.
        ix=(test_y==0).nonzero().flatten()[:2];z0,z1=test_mu[ix]
        interpolated=torch.stack([models['vae'].decode(((1-t)*z0+t*z1)[None],torch.zeros(1,dtype=torch.long,device=device))[0].cpu() for t in torch.linspace(0,1,9)])
    # Fit a 2D PCA projection only on training latents, then project test points.
    with torch.no_grad():
        train_mu,_=models['vae'].encode(x,y);center=train_mu.mean(0)
        _,_,basis=torch.linalg.svd(train_mu-center,full_matrices=False)
        projection=((test_mu-center)@basis[:2].T).cpu().numpy()
    summary={'dataset':'scikit-learn digits (not MNIST), 8x8 grayscale',
        'train_size':len(x),'test_size':len(test),'split_seed':42,'training_seed_base':42,
        'device':device,'hardware':torch.cuda.get_device_name() if device.startswith('cuda') else platform.processor(),
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'training':training,'classifier_test_accuracy':holdout_acc,'vae_reconstruction_rmse':recon_rmse,
        'records':records,'scope':'small conditional educational implementations; not paper-scale reproductions'}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    web=ROOT/'src'/'data';web.mkdir(exist_ok=True)
    meta={**summary,'loss':logs,'reference':pixels(test[:100].cpu()),'reference_labels':test_y[:100].cpu().tolist(),
          'reconstruction':pixels(recon[:100].cpu()),'interpolation':pixels(interpolated),
          'latent_points':np.round(projection,4).tolist(),'latent_labels':test_y.cpu().tolist()}
    (web/'meta.js').write_text('window.GEN_META='+json.dumps(meta,separators=(',',':'))+';\nwindow.GEN_RUNS={};',encoding='utf-8')
    for key,value in gallery.items():
        (web/f'{key}.js').write_text('window.GEN_RUNS['+json.dumps(key)+']='+json.dumps(value,separators=(',',':'))+';',encoding='utf-8')
    print('EXPORT COMPLETE',flush=True)


if __name__=='__main__':main()
