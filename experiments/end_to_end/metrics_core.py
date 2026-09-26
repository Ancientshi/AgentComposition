"""Query-level ranking agreement; incomplete experiments cannot yield final summaries."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

RANKS=[1,2,3,4]


def average_ranks(values):
    out=[0.0]*len(values)
    ordered=sorted(range(len(values)),key=lambda i:values[i])
    for _,group in itertools.groupby(ordered,key=lambda i:values[i]):
        ids=list(group)
        positions=[ordered.index(i)+1 for i in ids]
        for i in ids:out[i]=sum(positions)/len(positions)
    return out


def correlation(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    x=x-x.mean();y=y-y.mean()
    den=math.sqrt(float((x*x).sum()*(y*y).sum()))
    return float((x*y).sum()/den) if den else None


def per_query(scores):
    assert len(scores)==4 and all(0<=x<=10 for x in scores)
    wins=losses=ties=0
    for i,j in itertools.combinations(range(4),2):
        wins+=scores[i]>scores[j];losses+=scores[i]<scores[j];ties+=scores[i]==scores[j]
    tau=(wins-losses)/math.sqrt(6*(wins+losses)) if wins+losses else None
    gains=np.exp2(np.asarray(scores,dtype=float))-1
    discount=1/np.log2(np.arange(2,6))
    ideal=float(np.dot(np.sort(gains)[::-1],discount))
    ndcg=float(np.dot(gains,discount)/ideal) if ideal else None
    return {'spearman':correlation(average_ranks([-r for r in RANKS]),average_ranks(scores)),
            'kendall_tau_b':tau,'pairwise_concordance':(wins+0.5*ties)/6,
            'pairwise_strict_accuracy':wins/6,'pairwise_tie_rate':ties/6,
            'ndcg4_among_executed_candidates':ndcg,
            'top_bottom_delta':scores[0]-scores[-1],
            'top_bottom_win':float(scores[0]>scores[-1]),'top_bottom_tie':float(scores[0]==scores[-1]),
            'top_bottom_loss':float(scores[0]<scores[-1]),
            **{f'score_position_{rank}':score for rank,score in zip(RANKS,scores)}}


def summarize(rows,seed=42,repetitions=10000):
    rng=np.random.default_rng(seed);n=len(rows)
    ids=rng.integers(0,n,size=(repetitions,n));result={'n_queries':n,'n_executions':4*n}
    for key in rows[0]['metrics']:
        data=np.array([r['metrics'][key] if r['metrics'][key] is not None else np.nan for r in rows],dtype=float)
        defined=int(np.isfinite(data).sum())
        if not defined:
            result[key]={'mean':None,'ci95':None,'defined_queries':0}
            if key in ['spearman','kendall_tau_b']:
                result[key].update(undefined_all_tied_queries=n,mean_with_undefined_set_to_zero=0.0)
            continue
        sampled=data[ids]
        denominator=np.isfinite(sampled).sum(axis=1)
        bootstrap=np.nansum(sampled,axis=1)[denominator>0]/denominator[denominator>0]
        result[key]={'mean':float(np.nanmean(data)),
                     'ci95':[float(x) for x in np.quantile(bootstrap,[.025,.975])] if defined>=2 else None,
                     'defined_queries':defined}
        if key in ['spearman','kendall_tau_b']:
            result[key]['undefined_all_tied_queries']=n-defined
            result[key]['mean_with_undefined_set_to_zero']=float(np.nan_to_num(data).mean())
    return result

