"""Local ResShift adapter. Upstream network and diffusion equations are unchanged."""
from pathlib import Path
import sys, math, time, importlib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/resshift'))
MODEL_NAME='ResShift realSR x4 · 4 steps · v3'
class Cancelled(Exception): pass

def instantiate(c):
 module,name=c.target.rsplit('.',1)
 return getattr(importlib.import_module(module),name)(**dict(c.params))

def efficient_attention(self,x):
 h=self.norm(x);b,c,hh,ww=h.shape
 q,k,v=[layer(h).flatten(2).transpose(1,2).unsqueeze(1) for layer in (self.q,self.k,self.v)]
 y=F.scaled_dot_product_attention(q,k,v).squeeze(1).transpose(1,2).reshape(b,c,hh,ww)
 return x+self.proj_out(y)

class Restorer:
 def __init__(self,device=None):
  self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
  self.model=None
 def load(self):
  if self.model is not None:return
  from ldm.modules.diffusionmodules.model import AttnBlock,MemoryEfficientAttnBlock
  # Equivalent single-head attention with PyTorch SDPA; avoids the full HW x HW matrix.
  AttnBlock.forward=efficient_attention;MemoryEfficientAttnBlock.forward=efficient_attention
  cfg=OmegaConf.load(ROOT/'vendor/resshift/configs/realsr_swinunet_realesrgan256_journal.yaml')
  self.steps=int(cfg.diffusion.params.steps)
  model=instantiate(cfg.model);ae=instantiate(cfg.autoencoder)
  for net,name in [(model,'resshift_realsrx4_s4_v3.pth'),(ae,'autoencoder_vq_f4.pth')]:
   state=torch.load(ROOT/'weights'/name,map_location='cpu',weights_only=True)
   if 'state_dict' in state:state=state['state_dict']
   for prefix in ('module.','_orig_mod.'):
    if all(k.startswith(prefix) for k in state):state={k[len(prefix):]:v for k,v in state.items()}
   net.load_state_dict(state,strict=True)
  self.diffusion=instantiate(cfg.diffusion)
  self.ae=ae.eval().to(self.device)
  self.model=model.eval().to(self.device)
  if self.device.type=='cuda':
   self.ae.half();self.model.dtype=torch.float16;self.model.convert_to_fp16()
  self.tile=128 if self.device.type=='cuda' else 64
 @torch.inference_mode()
 def restore(self,image,seed=2026,progress=lambda a,b:None,cancel=lambda:False):
  self.load();start=time.perf_counter();arr=np.asarray(image.convert('RGB')).astype(np.float32)/127.5-1
  h,w=arr.shape[:2];tile=self.tile;stride=tile-32
  def starts(n):return list(range(0,n-tile,stride))+[max(0,n-tile)] if n>tile else [0]
  positions=[(y,x) for y in starts(h) for x in starts(w)]
  accum=np.zeros((h*4,w*4,3),np.float32);weight=np.zeros((h*4,w*4,1),np.float32)
  devices=[self.device.index or 0] if self.device.type=='cuda' else []
  with torch.random.fork_rng(devices=devices):
   torch.manual_seed(seed)
   for index,(y,x) in enumerate(positions):
    if cancel():raise Cancelled()
    patch=arr[y:y+tile,x:x+tile];ph,pw=patch.shape[:2]
    t=torch.from_numpy(patch.copy()).permute(2,0,1).unsqueeze(0).to(self.device)
    pad_h=math.ceil(ph/64)*64-ph;pad_w=math.ceil(pw/64)*64-pw
    t=F.pad(t,(0,pad_w,0,pad_h),mode='reflect' if ph>pad_h and pw>pad_w else 'replicate')
    final=None
    for step,out in enumerate(self.diffusion.p_sample_loop_progressive(y=t,model=self.model,first_stage_model=self.ae,clip_denoised=False,model_kwargs={'lq':t},progress=False)):
     if cancel():raise Cancelled()
     final=out['sample'];progress(index*self.steps+step+1,len(positions)*self.steps+1)
    restored=self.diffusion.decode_first_stage(final,self.ae).clamp(-1,1)[0,:,:ph*4,:pw*4]
    a=restored.float().cpu().permute(1,2,0).numpy()*.5+.5
    # Positive separable feather weights, also at external image edges.
    wh=np.minimum(np.arange(ph*4)+1,np.arange(ph*4,0,-1));ww=np.minimum(np.arange(pw*4)+1,np.arange(pw*4,0,-1))
    blend=(np.minimum(wh,64)[:,None]*np.minimum(ww,64)[None,:])[...,None].astype(np.float32)
    accum[y*4:(y+ph)*4,x*4:(x+pw)*4]+=a*blend;weight[y*4:(y+ph)*4,x*4:(x+pw)*4]+=blend
  output=np.rint(np.clip(accum/weight,0,1)*255).astype(np.uint8)
  progress(1,1)
  return Image.fromarray(output),{'model':MODEL_NAME,'seed':seed,'steps':self.steps,'tiles':len(positions),'tile_size':tile,'device':str(self.device),'inference_ms':round((time.perf_counter()-start)*1000),'input_size':[w,h],'output_size':[w*4,h*4]}
