"""Three disclosed demonstration cases, not a population benchmark."""
from pathlib import Path
import sys,json,io,hashlib
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from engine import Restorer

def psnr(a,b):
 x=np.asarray(a).astype(np.float64)/255;y=np.asarray(b).astype(np.float64)/255
 mse=float(np.mean((x-y)**2));return {'rmse':mse**.5,'psnr_db':None if mse==0 else float(-10*np.log10(mse))}
def main():
 restorer=Restorer();restorer.load();reports=[]
 for name in ['cat','coffee','astronaut']:
  folder=ROOT/'results/examples'/name
  source=Image.open(folder/'source.png').convert('RGB');source.thumbnail((384,384),Image.Resampling.LANCZOS)
  w=source.width//4*4;h=source.height//4*4;ref=source.resize((w,h),Image.Resampling.LANCZOS);ref.save(folder/'reference.png')
  low=ref.resize((w//4,h//4),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(.4))
  buf=io.BytesIO();low.save(buf,format='JPEG',quality=45,subsampling=2);low=Image.open(io.BytesIO(buf.getvalue())).convert('RGB');low.save(folder/'input.png')
  restored,meta=restorer.restore(low,2026);restored.save(folder/'output.png')
  base=low.resize(ref.size,Image.Resampling.BICUBIC);base.save(folder/'baseline.png')
  metrics={'bicubic':psnr(ref,base),'model':psnr(ref,restored)}
  report={**meta,'project':'PhotoRestore','case':name,'strength_percent':100,'reference_metrics':metrics,'degradation':{'downsample':4,'filter':'Pillow BICUBIC','gaussian_blur_radius':.4,'jpeg_quality':45,'jpeg_subsampling':2},'input_sha256':hashlib.sha256((folder/'input.png').read_bytes()).hexdigest(),'weights':json.loads((ROOT/'weights/manifest.json').read_text())}
  (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');reports.append(report)
  canvas=Image.new('RGB',(4*240,205),'#f6f5f1');d=ImageDraw.Draw(canvas)
  for i,(title,im) in enumerate([('Low resolution (enlarged)',low),('Bicubic interpolation',base),('ResShift diffusion',restored),('Reference',ref)]):
   d.text((i*240+10,12),title,fill='#28372a');tile=im.resize((224,160),Image.Resampling.BICUBIC);canvas.paste(tile,(i*240+8,37))
  canvas.save(folder/'comparison.png')
  print(name,metrics,meta['inference_ms'],flush=True)
 (ROOT/'results/evaluation.json').write_text(json.dumps({'scope':'3 demonstration images; one fixed degradation and seed; not a held-out benchmark','cases':reports},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
