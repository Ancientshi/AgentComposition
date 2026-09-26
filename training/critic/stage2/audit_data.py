"""Audit all prepared pair labels and summarize lengths without model calls."""
import argparse,collections,json
from pathlib import Path
from core import metrics,signature
from prepare import jsonl,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    summary={}
    for split in ['train','valid']:
        counts=collections.Counter();lengths={k:collections.Counter() for k in ['reference','candidate','preferred_redundancy','rejected_redundancy','preferred_required','rejected_required']}
        for r in jsonl(a.data/f'cases_{split}.jsonl'):
            cs=r['candidates'];gs=r['references'];counts['queries']+=1
            assert len({signature(c) for c in cs})==len(cs)
            counts['multiple_reference_queries']+=len(gs)>1
            counts['all_references_in_inventory']+=all(set(g['tools'])<=set(r['inventory']['tools']) and g['llm'] in r['inventory']['llms'] for g in gs)
            for g in gs:lengths['reference'][len(g['tools'])]+=1
            for c in cs:lengths['candidate'][len(c['tools'])]+=1
            for p in r['pairs']:
                x,y=cs[p['positive']],cs[p['negative']];X,Y=set(x['tools']),set(y['tools']);mx,my=metrics(x,gs),metrics(y,gs);kind=p['kind'];counts[kind]+=1
                if kind=='redundancy':
                    assert x['llm']==y['llm'] and X<Y
                    assert all(X&set(g['tools'])==Y&set(g['tools']) for g in gs)
                    assert abs(mx['cr']-my['cr'])<1e-12 and mx['cp']>my['cp']
                    lengths['preferred_redundancy'][len(X)]+=1;lengths['rejected_redundancy'][len(Y)]+=1
                elif kind=='missing_required':
                    assert x['llm']==y['llm'] and Y<X and mx['cr']>my['cr'] and mx['cp']>=my['cp']-1e-12
                    lengths['preferred_required'][len(X)]+=1;lengths['rejected_required'][len(Y)]+=1
                elif kind=='replacement':
                    assert x['llm']==y['llm'] and len(X)==len(Y) and mx['cp']>my['cp'] and mx['cr']>=my['cr']-1e-12
                elif kind=='teacher_llm':assert x['llm']!=y['llm'] and X==Y
                else:raise ValueError(kind)
        summary[split]={'counts':dict(counts),'length_distribution':{k:dict(sorted(v.items())) for k,v in lengths.items()},'data_sha256':sha(a.data/f'cases_{split}.jsonl')}
    a.output.write_text(json.dumps({'all_pair_invariants_passed':True,'splits':summary},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
