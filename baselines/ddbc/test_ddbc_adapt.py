"""Mechanism/protocol checks; synthetic fixtures are never reported as results."""
import os,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import baseline5_run_infer_rag_gpt as b5
b5.ensure_env_cuda_library()
import torch
from run_ddbc_adapt import build_model,sample_sets,load_module

def fixture():
    torch.manual_seed(7)
    m=build_model(torch,np.random.RandomState(7).normal(size=(16,8)).astype('float32')).eval()
    return m,torch.randn(1,8)

class Checks(unittest.TestCase):
    def test_permutation_equivariance_and_set_context(self):
        m,q=fixture(); x=torch.tensor([[2,16,4,16]]); t=torch.tensor([.7]); le=torch.tensor([4])
        with torch.no_grad():
            a=m(x,q,t,le); order=torch.tensor([2,0,3,1]); b=m(x[:,order],q,t,le)
            torch.testing.assert_close(a[:,order],b,atol=2e-6,rtol=2e-5)
            c=m(torch.tensor([[3,16,4,16]]),q,t,le)
            self.assertGreater((a[:,1]-c[:,1]).abs().max().item(),1e-6)
            self.assertGreater((a-m(x,q+1,t,le)).abs().max().item(),1e-6)
    def test_sampler_uniqueness_completion_replay(self):
        m,q=fixture()
        with torch.no_grad():
            a,s=sample_sets(torch,m,q[0],[0.,0.,0.,1.],4,12,51)
            b,t=sample_sets(torch,m,q[0],[0.,0.,0.,1.],4,12,51)
        self.assertEqual(a,b); self.assertEqual(s,t)
        self.assertTrue(all(len(x)==4 and len(set(x))==4 and max(x)<16 for x in a))
        self.assertTrue(np.isfinite(s).all())
    def test_official_noise_weight(self):
        noise=load_module(Path(__file__).parent/'upstream/noise_schedule.py','test_noise').LogLinearNoise()
        t=torch.tensor([.001,.2,.7,1.]); sigma,dsigma=noise(t)
        torch.testing.assert_close(-torch.expm1(-sigma),.999*t)
        torch.testing.assert_close(dsigma/torch.expm1(sigma),1/t)
    def test_training_loss_has_finite_gradient(self):
        m,q=fixture(); m.train(); x=torch.tensor([[2,16,4,16]])
        logits=m(x,q,torch.tensor([.7]),torch.tensor([4]))
        loss=torch.nn.functional.cross_entropy(logits[:,[1,3]].reshape(-1,16),torch.tensor([5,7]))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

if __name__=='__main__':
    torch.set_num_threads(2); unittest.main()
