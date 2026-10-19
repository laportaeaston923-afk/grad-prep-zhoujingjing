import sys
import threading
import time
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src'/'ml'))
from model import RepairNet,repair,make_masks
from engine import scores
from server import create_app,validate

class StubEngine:
    def health(self):return {'ready':True,'device':'test'}
    def run(self,payload,callback=None):
        for i in range(5):
            time.sleep(.01)
            if callback:callback(i+1,5)
        return {'has_reference':'reference' in payload,'input_holes_zero':all(v==0 for v,m in zip(payload['pixels'],payload['mask']) if m)}

class RepairChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def payload(self):
        mask=make_masks(1,kind='center')[0,0].astype(int).reshape(-1).tolist()
        return {'pixels':[123]*784,'mask':mask,'sampler':'ddim','steps':10,'candidates':1,'seed':7}
    def test_request_validation(self):
        p=self.payload();clean=validate(p)
        self.assertTrue(all(v==0 for v,m in zip(clean['pixels'],clean['mask']) if m))
        for change in [{'mask':[0]*784},{'mask':[1]*784},{'sampler':'invalid'},{'sampler':'ddpm','steps':20},{'candidates':9},{'pixels':[1.5]*784},{'seed':True}]:
            with self.assertRaises(ValueError):validate({**p,**change})
        with self.assertRaises(ValueError):validate({**p,'reference':[124]*784})
    def test_sampler_preserves_known_pixels_and_ignores_hidden_values(self):
        torch.manual_seed(42);model=RepairNet().eval();image=torch.randn(1,1,28,28).clamp(-1,1);hole=torch.tensor(make_masks(1,kind='center'))
        a,frames=repair(model,image,hole,steps=10,seeds=[7]);modified=image*(1-hole)+torch.randn_like(image)*hole
        b,_=repair(model,modified,hole,steps=10,seeds=[7]);torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(a*(1-hole),image*(1-hole),rtol=0,atol=0)
        self.assertTrue(torch.isfinite(a).all());self.assertEqual(frames[-1][0],10)
    def test_metrics_only_score_holes(self):
        mask=[1]+[0]*783;truth=[0]*784;output=[255]*784
        self.assertEqual(scores(output,truth,mask)['masked_rmse'],1)
        output[0]=0;self.assertEqual(scores(output,truth,mask)['masked_rmse'],0)
        self.assertIsNone(scores(output,truth,mask)['masked_ink_rmse'])
    def test_async_api_and_cancel(self):
        app=create_app(StubEngine());client=app.test_client()
        self.assertTrue(client.get('/api/health').json['ready'])
        bad=client.post('/api/jobs',json={});self.assertEqual(bad.status_code,422)
        submitted=client.post('/api/jobs',json=self.payload());self.assertEqual(submitted.status_code,202)
        job=submitted.json['id']
        for _ in range(100):
            state=client.get('/api/jobs/'+job).json
            if state['status']=='done':break
            time.sleep(.01)
        self.assertEqual(state['status'],'done');self.assertTrue(state['result']['input_holes_zero']);self.assertFalse(state['result']['has_reference'])
        job=client.post('/api/jobs',json=self.payload()).json['id'];client.delete('/api/jobs/'+job)
        for _ in range(100):
            state=client.get('/api/jobs/'+job).json
            if state['status']=='cancelled':break
            time.sleep(.01)
        self.assertEqual(state['status'],'cancelled');self.assertEqual(client.get('/api/jobs/missing').status_code,404)
        app.extensions['repair_worker'].shutdown(wait=True)

if __name__=='__main__':unittest.main()
