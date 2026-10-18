import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'ml'))
from models import MLP,VAE,TinyDiT,schedule,diffusion,flow_sample


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_dataset_split(self):
        d=np.load(Path(__file__).resolve().parents[1]/'results'/'dataset.npz')
        self.assertEqual(d['train'].shape,(1437,64))
        self.assertEqual(d['test'].shape,(360,64))
        self.assertFalse(set(d['train_indices']) & set(d['test_indices']))
        self.assertTrue(np.isin(d['train_labels'],np.arange(10)).all())

    def test_schedule(self):
        beta,abar=schedule()
        self.assertTrue(torch.all((beta>0)&(beta<1)))
        self.assertTrue(torch.all(abar[1:]<abar[:-1]))
        self.assertLess(float(abar[-1]),1e-5)

    def test_dit_and_vae_shapes(self):
        x=torch.randn(2,64);t=torch.rand(2);y=torch.tensor([0,9])
        self.assertEqual(TinyDiT()(x,t,y).shape,x.shape)
        recon,mu,lv=VAE()(x,y)
        self.assertEqual(recon.shape,x.shape)
        self.assertEqual(mu.shape,(2,12))
        self.assertTrue(torch.isfinite(recon).all())

    def test_ddim_determinism_and_limits(self):
        model=MLP().eval();noise=torch.randn(2,64);y=torch.tensor([2,7])
        a,frames=diffusion(model,y,noise,steps=5)
        b,_=diffusion(model,y,noise,steps=5)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(len(frames),6)
        self.assertTrue(torch.all(a.abs()<=1.00001))
        with self.assertRaises(ValueError):diffusion(model,y,noise,steps=5,stochastic=True)

    def test_meanflow_jvp_gradient(self):
        model=MLP(meanflow=True);z=torch.randn(2,64);v=torch.randn_like(z)
        t=torch.tensor([.5,.8]);r=torch.tensor([.2,.4]);y=torch.tensor([1,3])
        u,du=torch.func.jvp(lambda z,t,r:model(z,t,y,r),(z,t,r),(v,torch.ones_like(t),torch.zeros_like(r)))
        target=(v-(t-r)[:,None]*du).detach();loss=(u-target).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)
        out,frames=flow_sample(model,y,z,steps=1,meanflow=True)
        self.assertEqual(len(frames),2)
        self.assertTrue(torch.isfinite(out).all())


if __name__=='__main__':unittest.main()
