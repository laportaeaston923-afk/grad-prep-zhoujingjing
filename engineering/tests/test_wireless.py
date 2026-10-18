import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'/'wireless'))
from model import simulate, schedule, sample, Denoiser


class WirelessChecks(unittest.TestCase):
    def test_simulation(self):
        a,b = simulate(4,123),simulate(4,123)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
        x,y,m,f = a
        self.assertEqual(x.shape,(4,8,128))
        self.assertTrue(np.isfinite(y).all())
        self.assertTrue((y[m==0]==0).all())
        self.assertFalse(np.array_equal(x,simulate(4,456)[0]))

    def test_schedule(self):
        beta,alpha,abar,var = schedule()
        self.assertTrue(torch.all((beta>0)&(beta<1)))
        self.assertTrue(torch.all(abar[1:]<abar[:-1]))
        self.assertEqual(float(var[0]),0)
        self.assertTrue(torch.all(var>=0))
        self.assertLess(float(abar[-1]),1e-5)

    def test_seeded_sampling(self):
        torch.set_num_threads(2)
        model = Denoiser()
        _,y,m,_ = simulate(1,321)
        y,m = torch.tensor(y),torch.tensor(m)
        a,frames=sample(model,y,m,seed=7,keep_frames=True)
        b,_=sample(model,y,m,seed=7)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual([frames[0][0],frames[-1][0],len(frames)],[100,0,21])


if __name__=='__main__': unittest.main()
