"""Small conditional implementations; not full-scale paper reproductions."""
import math
import torch
from torch import nn

T = 100

class TimeEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.register_buffer('freq', torch.exp(-math.log(10000)*torch.arange(dim//2)/(dim//2-1)))
    def forward(self,t):
        angle=t[:,None]*100*self.freq
        return torch.cat([angle.sin(),angle.cos()],-1)


class MLP(nn.Module):
    def __init__(self, dim=64, meanflow=False):
        super().__init__()
        self.meanflow=meanflow
        self.time=TimeEmbedding()
        self.label=nn.Embedding(10,16)
        self.net=nn.Sequential(nn.Linear(dim+32+16+(32 if meanflow else 0),256),nn.SiLU(),
                               nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,dim))
    def forward(self,x,t,y,r=None):
        parts=[x,self.time(t),self.label(y)]
        if self.meanflow: parts.append(self.time(t-r))
        return self.net(torch.cat(parts,-1))


class VAE(nn.Module):
    latent=12
    def __init__(self):
        super().__init__()
        self.label=nn.Embedding(10,16)
        self.encoder=nn.Sequential(nn.Linear(80,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.mu=nn.Linear(128,self.latent);self.logvar=nn.Linear(128,self.latent)
        self.decoder=nn.Sequential(nn.Linear(self.latent+16,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,64),nn.Tanh())
    def encode(self,x,y):
        h=self.encoder(torch.cat([x,self.label(y)],-1))
        return self.mu(h),self.logvar(h).clamp(-8,4)
    def decode(self,z,y):return self.decoder(torch.cat([z,self.label(y)],-1))
    def forward(self,x,y):
        mu,lv=self.encode(x,y)
        return self.decode(mu+torch.exp(.5*lv)*torch.randn_like(mu),y),mu,lv


class DiTBlock(nn.Module):
    def __init__(self,dim=64):
        super().__init__()
        self.norm1=nn.LayerNorm(dim,elementwise_affine=False)
        self.norm2=nn.LayerNorm(dim,elementwise_affine=False)
        self.attn=nn.MultiheadAttention(dim,4,batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
        self.ada=nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        nn.init.zeros_(self.ada[-1].weight);nn.init.zeros_(self.ada[-1].bias)
    def forward(self,x,c):
        shift1,scale1,gate1,shift2,scale2,gate2=self.ada(c).chunk(6,-1)
        z=self.norm1(x)*(1+scale1[:,None])+shift1[:,None]
        x=x+gate1[:,None]*self.attn(z,z,z,need_weights=False)[0]
        z=self.norm2(x)*(1+scale2[:,None])+shift2[:,None]
        return x+gate2[:,None]*self.mlp(z)


class TinyDiT(nn.Module):
    """Pixel-space DiT-inspired backbone: 2x2 patches, adaLN-Zero, 3 blocks."""
    def __init__(self):
        super().__init__()
        self.time=nn.Sequential(TimeEmbedding(64),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,64))
        self.label=nn.Embedding(10,64)
        self.patch=nn.Linear(4,64)
        self.position=nn.Parameter(torch.randn(1,16,64)*.02)
        self.blocks=nn.ModuleList([DiTBlock() for _ in range(3)])
        self.final=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,4))
    def forward(self,x,t,y):
        patches=x.reshape(-1,4,2,4,2).permute(0,1,3,2,4).reshape(-1,16,4)
        z=self.patch(patches)+self.position;c=self.time(t)+self.label(y)
        for block in self.blocks:z=block(z,c)
        return self.final(z).reshape(-1,4,4,2,2).permute(0,1,3,2,4).reshape(-1,64)


def schedule(device='cpu'):
    f=torch.cos((torch.linspace(0,1,T+1,device=device)+.008)/1.008*math.pi/2)**2
    beta=(1-f[1:]/f[:-1]).clamp(max=.999)
    abar=(1-beta).cumprod(0)
    return beta,abar


@torch.no_grad()
def diffusion(model,y,noise,steps=100,stochastic=False,latent=False,keep_frames=True):
    if stochastic and steps!=T:raise ValueError('Ancestral DDPM requires all 100 steps in this implementation')
    beta,abar=schedule(noise.device);x=noise.clone();frames=[x.cpu()] if keep_frames else []
    times=torch.linspace(T-1,0,steps,device=x.device).round().long()
    for k,t in enumerate(times):
        ti=int(t);prev=int(times[k+1]) if k+1<len(times) else -1
        a=abar[ti];ap=abar[prev] if prev>=0 else torch.ones((),device=x.device)
        eps=model(x,torch.full((len(x),),ti/T,device=x.device),y)
        clean=(x-(1-a).sqrt()*eps)/a.sqrt()
        clean=clean.clamp(-4,4) if latent else clean.clamp(-1,1)
        if stochastic:
            variance=beta[ti]*(1-ap)/(1-a)
            mean=beta[ti]*ap.sqrt()/(1-a)*clean+(1-ap)*(1-beta[ti]).sqrt()/(1-a)*x
            x=mean+(variance.sqrt()*torch.randn_like(x) if ti else 0)
        else:
            # Recompute epsilon after x0 clipping, eta=0 DDIM.
            corrected=(x-a.sqrt()*clean)/(1-a).sqrt()
            x=ap.sqrt()*clean+(1-ap).sqrt()*corrected
        if keep_frames:frames.append(x.cpu())
    return x,frames


@torch.no_grad()
def flow_sample(model,y,noise,steps=20,meanflow=False,keep_frames=True):
    x=noise.clone();frames=[x.cpu()] if keep_frames else []
    for k in range(steps):
        t=torch.full((len(x),),1-k/steps,device=x.device)
        r=torch.full_like(t,1-(k+1)/steps)
        velocity=model(x,t,y,r) if meanflow else model(x,t,y)
        x=x-velocity/steps
        if keep_frames:frames.append(x.cpu())
    return x,frames
