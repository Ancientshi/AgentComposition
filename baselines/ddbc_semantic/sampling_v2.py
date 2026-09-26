"""Feasibility-preserving masked diffusion; observed tokens are never remasked."""
import collections,math
import numpy as np
from guided import LegalCodes

def has_distinct_assignment(domains):
    """Exact bipartite feasibility for at most six tool slots.

Only domains with fewer than K elements need matching: once these are matched,
any remaining domain of size >= K can avoid the at most K-1 occupied IDs.
"""
    k=len(domains)
    if any(len(d)==0 for d in domains):return False
    small=sorted([d for d in domains if len(d)<k],key=len);owners={}
    def augment(i,seen):
        for item in small[i]:
            item=int(item)
            if item in seen:continue
            seen.add(item)
            if item not in owners or augment(owners[item],seen):owners[item]=i;return True
        return False
    return all(augment(i,set()) for i in range(len(small)))

def draw_bundles(model,query,codes,catalog,lengths,steps,seed,component_prior,schedule='constant'):
    import torch
    rng=np.random.default_rng(seed);legality=LegalCodes(codes,catalog);device=query.device;proposals=[];stats=collections.Counter()
    for k in sorted(set(lengths)):
        n=lengths.count(k);x=model.empty(n,k,device).cpu().numpy();partial=np.full((n,k+1,4),-1,dtype=np.int64);score=np.zeros(n)
        domains=[[legality.pool['llm']]+[legality.pool['tool'] for _ in range(k)] for _ in range(n)]
        for step in range(steps,0,-1):
            raw=model(torch.tensor(x,device=device),query[None].expand(n,-1),torch.full((n,),-math.log(1-.999*step/steps),device=device)).cpu().numpy()
            reveals=(partial<0)&(rng.random(partial.shape)<1/step)
            beta=1. if schedule=='constant' else .25+.75*(steps-step)/max(steps-1,1)
            for b in range(n):
                positions=np.argwhere(reveals[b]);rng.shuffle(positions)
                for j,l in positions:
                    ids=domains[b][j];values=np.unique(codes[ids,l]);groups=[];valid=[]
                    for v in values:
                        subset=ids[codes[ids,l]==v]
                        if j and len(subset)<k:
                            trial=domains[b][1:].copy();trial[j-1]=subset
                            if not has_distinct_assignment(trial):stats['infeasible_code_values_pruned']+=1;continue
                        valid.append(v);groups.append(subset)
                    assert valid,'Feasible state must have a feasible refinement'
                    values=np.array(valid);pos=2+5*j+l
                    logits=raw[b,pos,values+model.offsets[l]].astype(np.float64)
                    masses=np.array([component_prior[g].sum(dtype=np.float64) for g in groups]);logits+=beta*np.log(np.maximum(masses,1e-30))
                    probs=np.exp(logits-logits.max());probs/=probs.sum();choice=int(rng.choice(len(values),p=probs));v=int(values[choice])
                    partial[b,j,l]=v;x[b,pos]=v+model.offsets[l];domains[b][j]=groups[choice];score[b]+=math.log(max(probs[choice],1e-300))
        for b in range(n):
            assert np.all(partial[b]>=0);assert all(len(d)==1 for d in domains[b]);ids=[int(d[0]) for d in domains[b]];assert len(set(ids[1:]))==k
            proposals.append({'components':[ids[0]]+sorted(ids[1:]),'reverse_path_logp':float(score[b])})
    return proposals,dict(stats)
