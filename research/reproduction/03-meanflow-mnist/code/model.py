"""MNIST MeanFlow / iMF method-level reproduction. One u-head, JVP, one-step sampling."""
import math
import torch
from torch import nn
import torch.nn.functional as F

def time_embed(x, dim):
    half=dim//2
    k=torch.exp(-math.log(10000)*torch.arange(half,device=x.device,dtype=x.dtype)/max(half-1,1))
    y=x[:,None]*k[None,:]*1000
    return torch.cat([y.sin(),y.cos()],dim=1)

class ResBlock(nn.Module):
    def __init__(self,cin,cout,dim):
        super().__init__()
        self.n1=nn.GroupNorm(8,cin)
        self.c1=nn.Conv2d(cin,cout,3,padding=1)
        self.n2=nn.GroupNorm(8,cout)
        self.c2=nn.Conv2d(cout,cout,3,padding=1)
        self.emb=nn.Linear(dim,cout)
        self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,emb):
        h=self.c1(F.silu(self.n1(x)))
        h=h+self.emb(F.silu(emb))[:,:,None,None]
        h=self.c2(F.silu(self.n2(h)))
        return (h+self.skip(x))/1.41421356237

class TinyVelocityUNet(nn.Module):
    """Single average-velocity head u; instantaneous v=u(z,t,t), Eq.(12)."""
    def __init__(self,width=32,emb_dim=128):
        super().__init__()
        assert width%8==0
        self.width,self.emb_dim=width,emb_dim
        self.cond=nn.Sequential(nn.Linear(emb_dim,emb_dim*2),nn.SiLU(),nn.Linear(emb_dim*2,emb_dim))
        self.cls=nn.Embedding(11,emb_dim) # class 10 is null
        self.input=nn.Conv2d(1,width,3,padding=1)
        self.d0=ResBlock(width,width,emb_dim)
        self.down1=nn.Conv2d(width,width,3,stride=2,padding=1)
        self.d1=ResBlock(width,width*2,emb_dim)
        self.down2=nn.Conv2d(width*2,width*2,3,stride=2,padding=1)
        self.mid=ResBlock(width*2,width*2,emb_dim)
        self.up1=ResBlock(width*4,width*2,emb_dim)
        self.up0=ResBlock(width*3,width,emb_dim)
        self.output=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv2d(width,1,3,padding=1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)
    def forward(self,z,r,t,y,w):
        emb=time_embed(r,self.emb_dim)+time_embed(t,self.emb_dim)+time_embed(w,self.emb_dim)
        emb=self.cond(emb)+self.cls(y)
        h0=self.d0(self.input(z),emb)
        h1=self.d1(self.down1(h0),emb)
        h2=self.mid(self.down2(h1),emb)
        h=F.interpolate(h2,size=h1.shape[-2:],mode="nearest")
        h=self.up1(torch.cat([h,h1],dim=1),emb)
        h=F.interpolate(h,size=h0.shape[-2:],mode="nearest")
        h=self.up0(torch.cat([h,h0],dim=1),emb)
        return self.output(h)

@torch.no_grad()
def sample_one_step(model,labels,noise,cfg_scale=1.):
    """Exactly one model evaluation: z0=z1-u(z1,0,1)."""
    b=len(noise)
    r=torch.zeros(b,device=noise.device);t=torch.ones(b,device=noise.device)
    w=torch.full((b,),float(cfg_scale),device=noise.device)
    u=model(noise,r,t,labels,w)
    return noise-u,u

def loss_batch(model,x,labels,mode,p_diagonal=.5,cond_dropout=.1,max_cfg=4.):
    """MF JVP tangent=e-x; iMF JVP tangent=u(z,t,t). Plain v-loss."""
    assert mode in ("mf","imf","imf_cfg")
    b=len(x);device=x.device
    x=x.mul(2).sub(1);noise=torch.randn_like(x)
    pair=torch.randn(b,2,device=device).sub(.4).sigmoid()
    t=pair.max(dim=1).values;r=pair.min(dim=1).values
    diag=torch.rand(b,device=device)<p_diagonal
    r=torch.where(diag,t,r)
    delta=(t-r).view(b,1,1,1)
    z=(1-t[:,None,None,None])*x+t[:,None,None,None]*noise
    vc_true=noise-x
    w=torch.ones(b,device=device);c=labels
    if mode=="imf_cfg":
        w=torch.empty(b,device=device).uniform_(1.,max_cfg)
        dropout=torch.rand(b,device=device)<cond_dropout
        c=torch.where(dropout,torch.full_like(c,10),c)
        w=torch.where(dropout,torch.ones_like(w),w)
    with torch.no_grad():
        if mode=="mf":
            tangent_v=vc_true
            target_v=vc_true
        else:
            tangent_v=model(z,t,t,c,w)
            target_v=vc_true
            if mode=="imf_cfg":
                v_uncond=model(z,t,t,torch.full_like(c,10),torch.ones_like(w))
                target_v=vc_true+(1-1/w)[:,None,None,None]*(tangent_v-v_uncond)
        def fn(z_,r_,t_):
            return model(z_,r_,t_,c,w)
        _,dudt=torch.autograd.functional.jvp(fn,(z,r,t),
                     (tangent_v.detach(),torch.zeros_like(r),torch.ones_like(t)),
                     create_graph=False,strict=False)
    u=model(z,r,t,c,w)
    V=u+delta*dudt.detach()
    losses=(V-target_v.detach()).square().mean(dim=(1,2,3))
    loss=losses.mean();valid=~diag
    fig3=losses[valid].mean() if valid.any() else loss
    return loss,{"loss":float(loss.detach()),"fig3_offdiag_loss":float(fig3.detach()),
                 "offdiag_count":int(valid.sum())}
