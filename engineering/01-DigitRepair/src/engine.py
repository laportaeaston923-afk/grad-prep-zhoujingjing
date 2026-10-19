"""Inference engine: the reference image is used only by scoring, never by repair."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'ml'))
from model import RepairNet,repair,harmonic_fill

ROOT=Path(__file__).resolve().parents[1]

def scores(pixels,reference,hole):
    x=np.asarray(pixels,dtype=np.float64).reshape(28,28)/255
    y=np.asarray(reference,dtype=np.float64).reshape(28,28)/255
    mask=np.asarray(hole,dtype=bool).reshape(28,28)
    error=(x-y)[mask];mse=float(np.mean(error**2));ink=mask&(y>.2)
    return {'masked_rmse':float(np.sqrt(mse)),'masked_mae':float(np.mean(np.abs(error))),
            'masked_psnr_db':float(-10*np.log10(mse)) if mse>0 else None,
            'masked_ink_rmse':float(np.sqrt(np.mean((x[ink]-y[ink])**2))) if ink.any() else None}

class Engine:
    def __init__(self,device=None,checkpoint=None):
        torch.set_num_threads(4);self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint=Path(checkpoint) if checkpoint else ROOT/'results'/'repair.pt'
        self.model=None;self.training={}
        if self.checkpoint.exists():
            saved=torch.load(self.checkpoint,map_location=self.device,weights_only=True)
            self.model=RepairNet().to(self.device);self.model.load_state_dict(saved['weights']);self.model.eval();self.training=saved['training']
    def health(self):return {'ready':self.model is not None,'device':self.device,'image_size':28,'training':self.training}
    def run(self,payload,callback=None):
        if self.model is None:raise RuntimeError('模型权重不存在，请先训练或下载完整项目。')
        raw=np.asarray(payload['pixels'],dtype=np.uint8).reshape(28,28)
        mask=np.asarray(payload['mask'],dtype=np.float32).reshape(28,28)
        count=payload['candidates'];seeds=[payload['seed']+i for i in range(count)]
        image=torch.tensor(raw,device=self.device,dtype=torch.float32)[None,None]/127.5-1
        hole=torch.tensor(mask,device=self.device)[None,None]
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter()
        result,frames=repair(self.model,image.expand(count,-1,-1,-1),hole.expand(count,-1,-1,-1),steps=payload['steps'],sampler=payload['sampler'],seeds=seeds,callback=callback)
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        milliseconds=(time.perf_counter()-start)*1000
        baseline=harmonic_fill(image*(1-hole),hole)
        def encode(tensor):
            values=((tensor.detach().cpu().clamp(-1,1)+1)*127.5).round().byte().numpy().reshape(-1,28,28)
            for value in values:value[mask==0]=raw[mask==0]
            return values.reshape(-1,784).tolist()
        candidates=encode(result);base=encode(baseline)[0];reference=payload.get('reference')
        output={'sampler':payload['sampler'],'steps':payload['steps'],'seeds':seeds,'device':self.device,'elapsed_ms':round(milliseconds,2),
                'mask_fraction':float(mask.mean()),'candidates':[{'pixels':p,'seed':s,'metrics':scores(p,reference,mask) if reference is not None else None} for p,s in zip(candidates,seeds)],
                'baseline':{'name':'180 次四邻域调和插值','pixels':base,'metrics':scores(base,reference,mask) if reference is not None else None},
                'frames':[{'step':step,'candidates':encode(values)} for step,values in frames],
                'has_reference':reference is not None,'known_pixels_unchanged':True}
        return output
