"""Compact conditional U-Net and inpainting DDPM/DDIM samplers."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SIZE,T=28,100

class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('freq',torch.exp(-math.log(10000)*torch.arange(32)/31))
    def forward(self,t):
        a=t[:,None]*self.freq;return torch.cat([a.sin(),a.cos()],-1)

class Block(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.first=nn.Sequential(nn.GroupNorm(8,cin),nn.SiLU(),nn.Conv2d(cin,cout,3,padding=1))
        self.time=nn.Linear(96,cout)
        self.second=nn.Sequential(nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,padding=1))
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,t):
        z=self.first(x)+self.time(t)[:,:,None,None]
        return (self.second(z)+self.skip(x))/math.sqrt(2)

class RepairNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time=nn.Sequential(TimeEmbedding(),nn.Linear(64,96),nn.SiLU(),nn.Linear(96,96))
        self.input=nn.Conv2d(3,24,3,padding=1)
        self.enc1=Block(24,24);self.down1=nn.Conv2d(24,48,4,stride=2,padding=1)
        self.enc2=Block(48,48);self.down2=nn.Conv2d(48,72,4,stride=2,padding=1)
        self.mid1=Block(72,72);self.mid2=Block(72,72)
        self.up2=nn.Conv2d(72,48,3,padding=1);self.dec2=Block(96,48)
        self.up1=nn.Conv2d(48,24,3,padding=1);self.dec1=Block(48,24)
        self.output=nn.Sequential(nn.GroupNorm(8,24),nn.SiLU(),nn.Conv2d(24,1,3,padding=1))
    def forward(self,x,t,observed,hole):
        emb=self.time(t.float());z=self.input(torch.cat([x,observed,hole],1))
        h1=self.enc1(z,emb);h2=self.enc2(self.down1(h1),emb)
        z=self.mid2(self.mid1(self.down2(h2),emb),emb)
        z=self.up2(F.interpolate(z,size=h2.shape[-2:],mode='nearest'));z=self.dec2(torch.cat([z,h2],1),emb)
        z=self.up1(F.interpolate(z,size=h1.shape[-2:],mode='nearest'));z=self.dec1(torch.cat([z,h1],1),emb)
        return self.output(z)

def schedule(device='cpu'):
    f=torch.cos((torch.linspace(0,1,T+1,device=device)+.008)/1.008*math.pi/2)**2
    beta=(1-f[1:]/f[:-1]).clamp(max=.999);return beta,(1-beta).cumprod(0)

def make_masks(count,seed=None,rng=None,kind='mixed'):
    rng=rng or np.random.default_rng(seed);m=np.zeros((count,1,SIZE,SIZE),np.float32)
    for i in range(count):
        if kind=='center':m[i,0,7:21,7:21]=1;continue
        if kind=='half':m[i,0,:,14:]=1;continue
        if kind=='brush' or (kind=='mixed' and rng.random()<.4):
            for _ in range(rng.integers(2,5)):
                x,y=rng.uniform(4,24,2);angle=rng.uniform(0,2*np.pi);length=rng.uniform(5,20);radius=rng.uniform(1.5,3.5)
                for t in np.linspace(0,1,40):
                    xx=x+np.cos(angle)*length*(t-.5);yy=y+np.sin(angle)*length*(t-.5)
                    gy,gx=np.mgrid[:SIZE,:SIZE];m[i,0][(gx-xx)**2+(gy-yy)**2<radius**2]=1
        else:
            w,h=rng.integers(8,21,2);x=rng.integers(0,SIZE-w+1);y=rng.integers(0,SIZE-h+1);m[i,0,y:y+h,x:x+w]=1
    return m

@torch.no_grad()
def harmonic_fill(observed,hole,iterations=180):
    x=observed.clone();kernel=torch.tensor([[0,.25,0],[.25,0,.25],[0,.25,0]],dtype=x.dtype,device=x.device)[None,None]
    for _ in range(iterations):x=F.conv2d(F.pad(x,(1,1,1,1),mode='replicate'),kernel)*hole+observed*(1-hole)
    return x

@torch.no_grad()
def repair(model,image,hole,steps=30,sampler='ddim',seeds=None,callback=None,keep_frames=True):
    """image in [-1,1], hole=1 means missing. No hidden reference is accepted."""
    if sampler not in ['ddpm','ddim']:raise ValueError('Unknown sampler')
    if sampler=='ddpm' and steps!=T:raise ValueError('DDPM requires 100 steps')
    if not 2<=steps<=T:raise ValueError('steps must be 2..100')
    if seeds is None:seeds=list(range(len(image)))
    if len(seeds)!=len(image):raise ValueError('One seed per candidate')
    generators=[torch.Generator(device=image.device).manual_seed(int(s)) for s in seeds]
    def noise():return torch.stack([torch.randn(image.shape[1:],generator=g,device=image.device) for g in generators])
    observed=image*(1-hole);known_noise=noise();x=known_noise.clone();beta,abar=schedule(image.device)
    frames=[];times=torch.linspace(T-1,0,steps,device=image.device).round().long()
    for k,ti in enumerate(times.tolist()):
        a=abar[ti];prev=int(times[k+1]) if k+1<len(times) else -1;ap=abar[prev] if prev>=0 else torch.ones((),device=image.device)
        # Visible-state conditioning follows the same forward schedule.
        x=x*hole+(a.sqrt()*observed+(1-a).sqrt()*known_noise)*(1-hole)
        eps=model(x,torch.full((len(x),),ti,device=image.device),observed,hole)
        clean=((x-(1-a).sqrt()*eps)/a.sqrt()).clamp(-1,1)
        if sampler=='ddpm':
            variance=beta[ti]*(1-ap)/(1-a)
            x=beta[ti]*ap.sqrt()/(1-a)*clean+(1-ap)*(1-beta[ti]).sqrt()/(1-a)*x
            if prev>=0:x=x+variance.sqrt()*noise()
        else:
            corrected=(x-a.sqrt()*clean)/(1-a).sqrt();x=ap.sqrt()*clean+(1-ap).sqrt()*corrected
        visible=observed*(1-hole)+((x.clamp(-1,1)+1)/2*2-1)*hole
        if keep_frames and (k==0 or k==steps-1 or k%max(1,steps//10)==0):frames.append((k+1,visible.cpu()))
        if callback:callback(k+1,steps)
    result=x.clamp(-1,1)*hole+observed*(1-hole)
    return result,frames
