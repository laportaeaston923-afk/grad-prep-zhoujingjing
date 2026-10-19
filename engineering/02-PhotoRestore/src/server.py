"""PhotoRestore: loopback-only Flask app with one bounded inference worker."""
from pathlib import Path
import sys,io,time,json,uuid,threading,shutil,argparse,warnings
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageOps,UnidentifiedImageError
from flask import Flask,request,jsonify,send_from_directory,abort
from werkzeug.exceptions import HTTPException
from engine import Restorer,Cancelled,MODEL_NAME
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'runtime'
Image.MAX_IMAGE_PIXELS=20_000_000
app=Flask(__name__,static_folder=str(ROOT/'src/web'),static_url_path='')
app.config['MAX_CONTENT_LENGTH']=10*1024*1024
jobs={};lock=threading.RLock();worker=ThreadPoolExecutor(max_workers=1);restorer=Restorer()
MAX_JOBS=8;TTL=1800

def cleanup():
 now=time.time()
 for key,j in list(jobs.items()):
  if j['status'] in ('done','failed','cancelled') and (now-j['updated']>TTL or len(jobs)>=MAX_JOBS):
   folder=(RUNTIME/key).resolve()
   if folder.parent==RUNTIME.resolve() and folder.name==key and folder.exists():shutil.rmtree(folder)
   del jobs[key]

def read_image(raw,max_edge):
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('error',Image.DecompressionBombWarning)
   with Image.open(io.BytesIO(raw)) as source:
    if source.format not in ('PNG','JPEG','WEBP'):raise ValueError('请选择 PNG、JPEG 或 WebP 图片')
    if getattr(source,'n_frames',1)>1:raise ValueError('暂不支持动画图片')
    source.load();im=ImageOps.exif_transpose(source)
    original=im.size
    if max(original)<32:raise ValueError('图片长边至少为 32 像素')
    if im.mode in ('RGBA','LA') or 'transparency' in im.info:
     rgba=im.convert('RGBA');bg=Image.new('RGBA',rgba.size,'white');bg.alpha_composite(rgba);im=bg.convert('RGB')
    else:im=im.convert('RGB')
    im.thumbnail((max_edge,max_edge),Image.Resampling.LANCZOS)
    im.info.clear()
    return im.copy(),original
 except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
  raise ValueError('无法解码图片，或图片超过 2000 万像素') from exc

def integer(name,default,lo,hi,choices=None):
 raw=request.form.get(name,str(default))
 try:v=int(raw)
 except (ValueError,TypeError):raise ValueError('参数 '+name+' 必须是整数')
 if v<lo or v>hi or choices and v not in choices:raise ValueError('参数 '+name+' 超出范围')
 return v

def run_job(key):
 j=jobs[key]
 try:
  if j['cancel']:raise Cancelled()
  j.update(status='running',message='正在载入模型',updated=time.time())
  image=Image.open(RUNTIME/key/'input.png').convert('RGB')
  def progress(done,total):
   j.update(progress=done/total,message=f'扩散采样 {done}/{total}',updated=time.time())
  enhanced,meta=restorer.restore(image,j['seed'],progress,lambda:j['cancel'])
  if j['cancel']:raise Cancelled()
  base=image.resize(enhanced.size,Image.Resampling.BICUBIC)
  output=Image.blend(base,enhanced,j['strength']/100)
  base.save(RUNTIME/key/'baseline.png');output.save(RUNTIME/key/'output.png')
  report={**meta,'project':'PhotoRestore','original_size':j['original_size'],'input_sha256':j['input_sha256'],'strength_percent':j['strength'],'reference_metrics':None,'note':'无完整清晰参考图，不计算PSNR/SSIM。新增细节为模型推测。','weights':json.loads((ROOT/'weights/manifest.json').read_text())}
  (RUNTIME/key/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
  j.update(status='done',progress=1,message='处理完成',report=report,updated=time.time())
 except Cancelled:j.update(status='cancelled',message='任务已取消',updated=time.time())
 except Exception as exc:
  app.logger.exception('Restoration failed')
  if 'out of memory' in str(exc).lower():message='显存不足，请关闭其他显卡程序并选择较小处理尺寸'
  elif isinstance(exc,FileNotFoundError):message='缺少模型文件，请先运行 scripts/setup_models.py'
  else:message='模型推理失败，请查看服务窗口中的错误信息'
  j.update(status='failed',message=message,updated=time.time())

def public(j):
 return {k:v for k,v in j.items() if k not in ('cancel',)}

@app.get('/')
def index():return app.send_static_file('index.html')
@app.get('/api/health')
def health():
 import torch
 ready=all((ROOT/'weights'/n).exists() for n in ['resshift_realsrx4_s15_v1.pth','autoencoder_vq_f4.pth'])
 return jsonify(ready=ready,model=MODEL_NAME,device=str(restorer.device),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
@app.post('/api/jobs')
def create():
 import hashlib
 try:
  if 'image' not in request.files:raise ValueError('请上传一张图片')
  seed=integer('seed',2026,0,2147483640);strength=integer('strength',100,0,100);edge=integer('max_edge',384,192,512,[192,384,512])
  raw=request.files['image'].read()
  image,original=read_image(raw,edge)
 except ValueError as exc:return jsonify(error=str(exc)),400
 with lock:
  cleanup()
  if sum(j['status'] in ('queued','running') for j in jobs.values())>=3:return jsonify(error='任务队列已满，请稍后重试'),429
  key=uuid.uuid4().hex;folder=RUNTIME/key;folder.mkdir(parents=True)
  image.save(folder/'input.png')
  j=dict(id=key,status='queued',progress=0,message='等待处理',seed=seed,strength=strength,original_size=list(original),processing_size=list(image.size),input_sha256=hashlib.sha256(raw).hexdigest(),cancel=False,updated=time.time())
  jobs[key]=j;worker.submit(run_job,key)
  return jsonify(public(j)),202
@app.get('/api/jobs/<key>')
def status(key):
 with lock:
  cleanup()
  if key not in jobs:abort(404)
  return jsonify(public(jobs[key]))
@app.delete('/api/jobs/<key>')
def cancel(key):
 with lock:
  if key not in jobs:abort(404)
  j=jobs[key]
  if j['status'] in ('queued','running'):j['cancel']=True
  return jsonify(public(j))
@app.get('/api/jobs/<key>/<asset>')
def asset(key,asset):
 if key not in jobs or jobs[key]['status']!='done':abort(404)
 if asset not in ('input.png','baseline.png','output.png','report.json'):abort(404)
 return send_from_directory(RUNTIME/key,asset,as_attachment=request.args.get('download')=='1',download_name='photo-restore-'+asset)
@app.get('/examples/<path:name>')
def example(name):return send_from_directory(ROOT/'results/examples',name)
@app.errorhandler(HTTPException)
def http_error(exc):return jsonify(error='上传总大小请控制在 10 MB 内' if exc.code==413 else exc.description),exc.code
@app.after_request
def privacy(response):
 if request.path.startswith('/api/'):response.headers['Cache-Control']='no-store'
 return response
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8083);parser.add_argument('--device',choices=['cpu','cuda']);args=parser.parse_args()
 if args.device:restorer=Restorer(args.device)
 app.run(host='127.0.0.1',port=args.port,debug=False,threaded=True)
