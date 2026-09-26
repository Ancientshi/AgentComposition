#!/usr/bin/env python3
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import baseline5_run_infer_rag_gpt as b5
b5.ensure_env_cuda_library()
import torch
from model import RVQDiffusion,Noise,corrupt_loss,MASK,BOS,EOS,LLM_BOI,TOOL_BOI,OFFICIAL
from infer import LegalCodes,draw_bundles,rank_bundles

class SemanticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        self.centers=np.random.RandomState(7).normal(0,.05,(3,128,128)).astype('float32')
        self.model=RVQDiffusion(self.centers,[128,128,128,2],blocks=2).cuda()
        self.npcodes=np.array([[i//4,i%4,(i*3)%7,i%2] for i in range(12)],dtype='int64')
        self.codes=torch.tensor(self.npcodes,device='cuda');self.q=torch.randn(4,768,device='cuda')
        self.catalog=[{'id':i,'kind':'llm' if i<4 else 'tool'} for i in range(12)]
    def test_serialization_flags_and_code_domains(self):
        comps=torch.tensor([[0,4,6],[1,5,7]],device='cuda');x=self.model.serialize(comps,self.codes,permute=True)
        self.assertTrue((x[:,0]==BOS).all());self.assertTrue((x[:,-1]==EOS).all());self.assertTrue((x[:,1]==LLM_BOI).all())
        for pos in [6,11]:self.assertTrue((x[:,pos]==TOOL_BOI).all())
        z=self.model.code_logits(torch.zeros(2,x.shape[1],self.model.vocab,device='cuda'),2)
        for i in range(12):self.assertEqual(int(torch.isfinite(z[0,i]).sum()),self.model.code_sizes[i%4])
    def test_rvq_prefix_domains(self):
        legal=LegalCodes(self.npcodes,self.catalog)
        for c in self.catalog:
            code=self.npcodes[c['id']];self.assertEqual(legal.candidates(c['kind'],code).tolist(),[c['id']])
            for level in range(4):
                partial=np.full(4,-1);partial[level]=code[level]
                self.assertIn(c['id'],legal.candidates(c['kind'],partial))
    def test_finite_gradients_query_and_context(self):
        model=self.model;optimizer=torch.optim.Adam(model.parameters(),lr=.003);comps=torch.tensor([[0,4,5],[1,6,7],[2,8,9],[3,10,11]],device='cuda')
        for _ in range(3):
            model.train();loss,_=corrupt_loss(model,self.q,comps,self.codes,Noise())
            self.assertTrue(torch.isfinite(loss));optimizer.zero_grad();loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()));optimizer.step()
        model.eval();x=model.empty(4,2,'cuda');t=torch.ones(4,device='cuda')
        with torch.inference_mode():
            a=model(x,self.q,t);b=model(x,self.q.roll(1,0),t);self.assertGreater(float((a-b).abs().max()),1e-6)
            x[:,2]=5;c=model(x,self.q,t);self.assertGreater(float((a[:,7]-c[:,7]).abs().max()),1e-7)
    def test_sampler_replay_legality(self):
        self.model.eval()
        with torch.inference_mode():
            a,fa=draw_bundles(self.model,self.q[0],self.npcodes,self.catalog,[1,1,2,2,3,3],4,77)
            b,fb=draw_bundles(self.model,self.q[0],self.npcodes,self.catalog,[1,1,2,2,3,3],4,77)
        self.assertEqual(a,b);self.assertEqual(fa,fb);self.assertTrue(a)
        for p in a:
            ids=p['components'];self.assertLess(ids[0],4);self.assertTrue(all(i>=4 for i in ids[1:]));self.assertEqual(len(ids),len(set(ids)))
    def test_portable_attention_and_rope(self):
        qkv=torch.randn(2,5,3,8,16,device='cuda')
        rotary=OFFICIAL.Rotary(16).cuda();cos,sin=rotary(torch.empty(2,5,128,device='cuda'))
        rotated=OFFICIAL.apply_rotary_pos_emb(qkv,cos,sin)
        torch.testing.assert_close(rotated[:,:,2],qkv[:,:,2],atol=0,rtol=0)
        q,k,v=[z.transpose(1,2) for z in rotated.unbind(2)]
        explicit=(q@k.transpose(-1,-2)/4).softmax(-1)@v
        native=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
        torch.testing.assert_close(native,explicit,atol=2e-6,rtol=2e-5)
    def test_noise_equations(self):
        t=torch.tensor([.01,.3,.8,1.]);s,d=Noise()(t)
        torch.testing.assert_close(-torch.expm1(-s),.999*t);torch.testing.assert_close(d/torch.expm1(s),1/t)

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
