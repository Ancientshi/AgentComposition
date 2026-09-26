"""Official DDBC DDiT blocks, extended with query conditioning and typed boundaries."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
BASE=Path(__file__).resolve().parent

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
OFFICIAL=module(BASE/'upstream/models/dit_torch.py','semantic_ddbc_official_dit')
EMA=module(BASE/'upstream/models/ema.py','semantic_ddbc_official_ema').ExponentialMovingAverage
Noise=module(BASE/'upstream/noise_schedule.py','semantic_ddbc_official_noise').LogLinearNoise
BOS,LLM_BOI,TOOL_BOI,EOS,MASK=0,1,2,3,4

class RVQDiffusion(nn.Module):
    def __init__(self,centroids,code_sizes,hidden=128,blocks=6,heads=8,dropout=.1):
        super().__init__();self.code_sizes=list(code_sizes);self.offsets=np.cumsum([5]+list(code_sizes[:-1])).tolist();self.vocab=5+sum(code_sizes)
        self.embedding=nn.Embedding(self.vocab,hidden)
        nn.init.normal_(self.embedding.weight,std=.02)
        assert centroids.shape[2]==hidden
        with torch.no_grad():
            for l in range(3):self.embedding.weight[self.offsets[l]:self.offsets[l]+code_sizes[l]].copy_(torch.as_tensor(centroids[l]))
        self.time=OFFICIAL.TimestepEmbedder(hidden)
        self.query=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.query_token=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,hidden))
        self.rotary=OFFICIAL.Rotary(hidden//heads)
        self.blocks=nn.ModuleList([OFFICIAL.DDiTBlock(hidden,heads,hidden,dropout=dropout) for _ in range(blocks)])
        self.final=OFFICIAL.DDitFinalLayer(hidden,self.vocab,hidden)
        self.length_head=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,128),nn.SiLU(),nn.Dropout(dropout),nn.Linear(128,7))
    def forward(self,x,q,sigma):
        h=self.embedding(x);h=h.clone();h[:,0]+=self.query_token(q)
        c=F.silu(self.time(sigma)+self.query(q))
        rotary=self.rotary(h)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for b in self.blocks:h=b(h,rotary,c)
            out=self.final(h,c)
        return out.float()
    def length_logits(self,q):return self.length_head(q)
    def code_positions(self,k):return [2+5*j+l for j in range(k+1) for l in range(4)]
    def empty(self,n,k,device):
        x=torch.full((n,2+5*(k+1)),MASK,dtype=torch.long,device=device);x[:,0]=BOS;x[:,-1]=EOS
        x[:,1]=LLM_BOI
        for j in range(1,k+1):x[:,1+5*j]=TOOL_BOI
        return x
    def serialize(self,components,codes,permute=False):
        # components tensor: first LLM, then unordered tools; a batch has one tool count.
        n,k1=components.shape;k=k1-1;x=self.empty(n,k,components.device)
        if permute and k>1:
            order=torch.rand(n,k,device=components.device).argsort(1)
            components=torch.cat([components[:,:1],components[:,1:].gather(1,order)],1)
        values=codes[components]+torch.tensor(self.offsets,device=components.device)
        x[:,self.code_positions(k)]=values.flatten(1)
        return x
    def code_logits(self,raw,k):
        z=raw[:,self.code_positions(k),:];levels=torch.arange(z.shape[1],device=z.device)%4
        allowed=torch.zeros((4,self.vocab),device=z.device,dtype=torch.bool)
        for l,(start,size) in enumerate(zip(self.offsets,self.code_sizes)):allowed[l,start:start+size]=True
        return z.masked_fill(~allowed[levels][None],-torch.inf)


def corrupt_loss(model,q,components,codes,noise,t=None):
    x=model.serialize(components,codes,permute=model.training);n=len(x);k=components.shape[1]-1;pos=model.code_positions(k)
    if t is None:
        # Official DDBC antithetic time sampling.
        t=((torch.rand(n,device=x.device)+torch.arange(n,device=x.device))/n)*.999+.001
    sigma,dsigma=noise(t);mask=torch.rand((n,len(pos)),device=x.device)<(-torch.expm1(-sigma))[:,None]
    xt=x.clone();xt[:,pos]=x[:,pos].masked_fill(mask,MASK)
    z=model.code_logits(model(xt,q,sigma),k)
    ce=F.cross_entropy(z.transpose(1,2),x[:,pos],reduction='none')
    loss=((ce*mask).mean(1)*(dsigma/torch.expm1(sigma))).mean()
    length_loss=F.cross_entropy(model.length_logits(q),torch.full((n,),k,device=x.device,dtype=torch.long))
    return loss+.2*length_loss,{'diffusion':float(loss.detach()),'length':float(length_loss.detach()),'masked_ce':float((ce*mask).sum().detach()/mask.sum().clamp(min=1))}
