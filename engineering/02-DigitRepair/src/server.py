"""Local Flask API with one inference worker, bounded queue, polling and cancel."""
import argparse
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from flask import Flask,request,jsonify
from engine import Engine

WEB=Path(__file__).resolve().parent/'web'

def validate(data):
    if not isinstance(data,dict):raise ValueError('请求应为 JSON 对象')
    def array(key,binary=False):
        value=data.get(key)
        if not isinstance(value,list) or len(value)!=784:raise ValueError(key+' 必须包含 784 个像素')
        if any(type(v) is not int or not (0<=v<=(1 if binary else 255)) for v in value):raise ValueError(key+' 必须为合法整数像素')
        return value
    pixels=array('pixels');mask=array('mask',True);missing=sum(mask)
    if missing==0:raise ValueError('请先绘制需要修复的区域')
    if missing>int(784*.7):raise ValueError('缺损区域不能超过 70%，请保留足够的可见信息')
    sampler=data.get('sampler','ddim');steps=data.get('steps',30);count=data.get('candidates',2);seed=data.get('seed',2026)
    if sampler not in ['ddpm','ddim']:raise ValueError('仅支持 DDPM 或 DDIM')
    if type(steps) is not int or steps not in [10,20,30,50,100] or (sampler=='ddpm' and steps!=100):raise ValueError('采样步数不合法；DDPM 必须为 100 步')
    if type(count) is not int or not 1<=count<=4:raise ValueError('候选数量应为 1–4')
    if type(seed) is not int or not 0<=seed<=2147483640:raise ValueError('随机种子超出范围')
    result={'pixels':[v if not mask[i] else 0 for i,v in enumerate(pixels)],'mask':mask,'sampler':sampler,'steps':steps,'candidates':count,'seed':seed}
    if data.get('reference') is not None:
        reference=array('reference')
        if any(reference[i]!=pixels[i] for i in range(784) if not mask[i]):raise ValueError('参考图必须与可见区域一致')
        result['reference']=reference
    return result

def create_app(engine=None):
    app=Flask(__name__,static_folder=str(WEB),static_url_path='');app.config['MAX_CONTENT_LENGTH']=100*1024
    engine=engine or Engine();jobs={};lock=threading.Lock();worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='digit-repair')
    app.extensions['repair_engine']=engine;app.extensions['repair_jobs']=jobs;app.extensions['repair_worker']=worker
    @app.get('/')
    def index():return app.send_static_file('index.html')
    @app.get('/api/health')
    def health():return jsonify(engine.health())
    @app.post('/api/jobs')
    def create_job():
        try:payload=validate(request.get_json(silent=True))
        except ValueError as error:return jsonify(error=str(error)),422
        if not engine.health()['ready']:return jsonify(error='模型未就绪，请先准备权重。'),503
        with lock:
            stale=[key for key,j in jobs.items() if j['status'] in ['done','failed','cancelled'] and time.time()-j['created']>1800]
            for key in stale:jobs.pop(key,None)
            if sum(j['status'] in ['queued','running'] for j in jobs.values())>=4:return jsonify(error='任务队列已满，请稍后重试。'),429
            completed=[key for key,j in jobs.items() if j['status'] in ['done','failed','cancelled']]
            while len(jobs)>=24 and completed:jobs.pop(completed.pop(0))
            job_id=uuid.uuid4().hex;jobs[job_id]={'id':job_id,'status':'queued','progress':0,'created':time.time(),'cancel':False}
        def work():
            try:
                with lock:
                    if jobs[job_id]['cancel']:jobs[job_id]['status']='cancelled';return
                    jobs[job_id]['status']='running'
                def progress(step,total):
                    with lock:
                        if jobs[job_id]['cancel']:raise InterruptedError('用户已取消')
                        jobs[job_id]['progress']=round(step/total,4)
                result=engine.run(payload,callback=progress)
                with lock:
                    if jobs[job_id]['cancel']:jobs[job_id]['status']='cancelled'
                    else:jobs[job_id].update(status='done',progress=1,result=result)
            except InterruptedError:
                with lock:jobs[job_id]['status']='cancelled'
            except Exception as error:
                app.logger.exception('Repair task failed')
                with lock:jobs[job_id].update(status='failed',error='推理失败，请检查服务端日志。')
        worker.submit(work)
        return jsonify(id=job_id,status='queued'),202
    @app.get('/api/jobs/<job_id>')
    def job_status(job_id):
        with lock:
            job=jobs.get(job_id)
            if job is None:return jsonify(error='任务不存在或已过期'),404
            return jsonify({k:v for k,v in job.items() if k not in ['created','cancel']})
    @app.delete('/api/jobs/<job_id>')
    def cancel_job(job_id):
        with lock:
            if job_id not in jobs:return jsonify(error='任务不存在'),404
            job=jobs[job_id]
            if job['status'] in ['queued','running']:job['cancel']=True
            return jsonify(id=job_id,status=job['status'])
    @app.errorhandler(413)
    def too_large(error):return jsonify(error='请求过大'),413
    return app

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8082);parser.add_argument('--device',choices=['cpu','cuda']);args=parser.parse_args()
    create_app(Engine(device=args.device)).run(host='127.0.0.1',port=args.port,threaded=True,debug=False)
