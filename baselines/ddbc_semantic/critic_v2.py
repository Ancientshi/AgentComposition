"""Permutation-invariant, query-conditioned whole-bundle critic."""
import torch
from torch import nn
from torch.nn import functional as F

class BundleCritic(nn.Module):
    def __init__(self):
        super().__init__();self.query=nn.Linear(768,128);self.component=nn.Linear(768,128);self.interaction=nn.Linear(768,128);self.types=nn.Embedding(2,128)
        layer=nn.TransformerEncoderLayer(128,4,256,.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(522),nn.Linear(522,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,2))
    def forward(self,q,component_features,ids,probabilities):
        valid=ids>=0;safe=ids.clamp_min(0);c=component_features[safe];qp=self.query(q)
        typ=torch.ones_like(ids);typ[:,0]=0
        h=self.component(c)+qp[:,None]+self.interaction(c*q[:,None])+self.types(typ)
        h=self.encoder(h,src_key_padding_mask=~valid)
        num=valid.sum(1).clamp_min(1);mean=(h*valid[:,:,None]).sum(1)/num[:,None];mx=h.masked_fill(~valid[:,:,None],-1e9).max(1).values
        pp=probabilities.gather(1,safe)*valid;toolmask=valid.clone();toolmask[:,0]=False;k=toolmask.sum(1).clamp_min(1)
        tp=pp*toolmask;cos=(c*q[:,None]).sum(-1);pcos=(cos*toolmask).sum(1)/k
        # These summaries are all invariant to tool order.
        stats=torch.stack([pp[:,0],tp.sum(1),tp.max(1).values,tp.sum(1)/k,torch.log(pp[:,0].clamp_min(1e-9)),
                           (torch.log(pp.clamp_min(1e-9))*toolmask).sum(1)/k,(num-1)/6,cos[:,0],pcos,(cos.masked_fill(~toolmask,-1)).max(1).values],1)
        logits=self.head(torch.cat([qp,h[:,0],mean,mx,stats],1))
        return logits

def pad_components(bundles,device='cuda'):
    ids=torch.full((len(bundles),7),-1,dtype=torch.long,device=device)
    for i,b in enumerate(bundles):ids[i,:len(b)]=torch.tensor(b,device=device)
    return ids
