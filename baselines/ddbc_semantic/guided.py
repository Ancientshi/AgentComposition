#!/usr/bin/env python3
"""Query-conditional RVQ-code diffusion; test labels enter only final evaluation."""
import argparse,collections,json,math,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from prepare import sha,write

class LegalCodes:
    def __init__(self,codes,catalog):
        self.codes=codes;self.pool={kind:np.array([c['id'] for c in catalog if c['kind']==kind]) for kind in ['llm','tool']};self.cache={}
    def candidates(self,kind,partial):
        key=(kind,tuple(partial))
        if key not in self.cache:
            ix=self.pool[kind]
            for l,v in enumerate(partial):
                if v>=0:ix=ix[self.codes[ix,l]==v]
            self.cache[key]=ix
        return self.cache[key]


def draw_bundles(model,query,codes,catalog,lengths,steps,seed,component_prior=None,beta=0.):
    import torch
    from model import MASK
    rng=np.random.default_rng(seed);legality=LegalCodes(codes,catalog);device=query.device
    proposals=[];failures=collections.Counter()
    for k in sorted(set(lengths)):
        n=sum(x==k for x in lengths);x=model.empty(n,k,device).cpu().numpy();partial=np.full((n,k+1,4),-1,dtype=np.int64)
        invalid=np.zeros(n,dtype=bool);score=np.zeros(n);qq=query[None].expand(n,-1)
        for step in range(steps,0,-1):
            raw=model(torch.tensor(x,device=device),qq,torch.full((n,),-math.log(1-.999*step/steps),device=device)).cpu().numpy()
            reveals=(partial<0)&(rng.random(partial.shape)<1/step)&(~invalid[:,None,None])
            for b in range(n):
                if invalid[b]:continue
                positions=np.argwhere(reveals[b]);rng.shuffle(positions)
                for j,l in positions:
                    kind='llm' if j==0 else 'tool';ids=legality.candidates(kind,partial[b,j])
                    if j>0:
                        used=set()
                        for jj in range(1,k+1):
                            if jj!=j and np.all(partial[b,jj]>=0):used.update(legality.candidates('tool',partial[b,jj]).tolist())
                        if used:ids=ids[~np.isin(ids,list(used))]
                    if not len(ids):invalid[b]=True;failures['constraint_dead_end']+=1;break
                    values=np.unique(codes[ids,l]);pos=2+5*j+l;tokenids=values+model.offsets[l]
                    logits=raw[b,pos,tokenids].astype(np.float64)
                    if beta:
                        assert component_prior is not None
                        masses=np.array([component_prior[ids[codes[ids,l]==v]].sum(dtype=np.float64) for v in values])
                        logits+=beta*np.log(np.maximum(masses,1e-30))
                    probs=np.exp(logits-logits.max());probs/=probs.sum()
                    choice=int(rng.choice(len(values),p=probs));v=values[choice]
                    partial[b,j,l]=v;x[b,pos]=v+model.offsets[l];score[b]+=math.log(max(probs[choice],1e-300))
        for b in range(n):
            if invalid[b]:continue
            if np.any(partial[b]<0):raise AssertionError('Unresolved MASK')
            ids=[]
            for j in range(k+1):
                options=legality.candidates('llm' if j==0 else 'tool',partial[b,j]);assert len(options)==1;ids.append(int(options[0]))
            assert len(set(ids[1:]))==k
            proposals.append({'components':[ids[0]]+sorted(ids[1:]),'reverse_path_logp':float(score[b])})
    return proposals,dict(failures)

