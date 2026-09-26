#!/usr/bin/env python3
"""Independent stdlib-only audit of saved predictions and project metrics."""
import argparse,collections,hashlib,json,math,re
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(path):
    read=lambda name:json.loads((path/name).read_text())
    cat=read('catalog.json'); by_id={a['id']:a for a in cat}
    rows=[json.loads(s) for s in (path/'results.jsonl').open()]
    blind=[json.loads(s) for s in (path/'predictions_blind.jsonl').open()]
    expected=read('evaluation/top10_metrics.json')['mean']; metrics=collections.defaultdict(list)
    max_tools=read('config.json').get('max_tools')
    if max_tools is not None:
        pool=read('candidate_pool.json')
        assert pool==[a for a in cat if len(a['tool_tokens'])<=max_tools]
    assert len(rows)==len(blind)==100
    assert len({r['dataset_example']['sample_id'] for r in rows})==100
    fingerprints=set()
    for a in cat:
        key=(a['llm_token'],tuple(sorted(a['tool_tokens'])))
        assert key not in fingerprints; fingerprints.add(key)
    for r,p in zip(rows,blind):
        assert p['sample_id']==r['dataset_example']['sample_id']
        assert p['results']==r['results']; assert len(r['results'])==10
        assert len({a['id'] for a in r['results']})==10
        for b in p['sampled_agent_sets']:
            assert 1<=len(b)<=5 and len(set(b))==len(b)
            assert all(i in by_id for i in b)
        frequency=collections.Counter(i for b in p['sampled_agent_sets'] for i in b)
        for a in r['results']:
            if max_tools is not None: assert len(a['tool_tokens'])<=max_tools
            assert a['llm_token']==by_id[a['id']]['llm_token']
            assert a['tool_tokens']==by_id[a['id']]['tool_tokens']
            assert a['inclusion_frequency']==frequency[a['id']]/len(p['sampled_agent_sets'])
        gold=r['dataset_example']['target'].split('<SPECIAL_END>')[0]
        llm=re.findall(r'<LLM_[^<>\n\r]+>',gold)[0]
        gt=set(re.findall(r'<<[^<>\n\r]+>>|<TOOL_[^<>\n\r]+>',gold)) - {'<TOOL_SEP>','<TOOL_EMPTY>'}
        gc=gt|{llm}; cr=[]; th=[]; hit=[]
        for j,a in enumerate(r['results']):
            pt=set(a['tool_tokens']); intersection=len(pt&gt)
            recall=intersection/len(gt) if gt else 1.
            precision=intersection/len(pt) if pt else float(not gt)
            cr.append(len((pt|{a['llm_token']})&gc)/len(gc))
            th.append(gt<=pt); hit.append(gt<=pt and llm==a['llm_token'])
            if j==0:
                metrics['top1_tool_recall'].append(recall); metrics['top1_tool_precision'].append(precision)
                metrics['top1_tool_f1'].append(2*precision*recall/(precision+recall) if precision+recall else 0.)
        metrics['top1_component_recall'].append(cr[0]); metrics['tool_hit@10'].append(float(any(th)))
        metrics['cr_hit@1'].append(float(hit[0])); metrics['cr_hit@10'].append(float(any(hit)))
        metrics['cr_mrr@10'].append(next((1/(i+1) for i,v in enumerate(hit) if v),0.))
        weights=[1/math.log2(i+2) for i in range(10)]
        metrics['rdcr@10'].append(sum(a*b for a,b in zip(cr,weights))/sum(weights))
    actual={k:sum(v)/len(v) for k,v in metrics.items()}
    assert all(abs(actual[k]-expected[k])<1e-12 for k in expected)
    complete=read('completed.json')
    assert sha(path/'predictions_blind.jsonl')==complete['predictions_sha256']
    assert sha(path/'results.jsonl')==complete['results_sha256']
    output={'passed':True,'n':100,'all_top10_unique':True,'all_outputs_in_training_catalog':True,
            'all_sampled_sets_valid':True,'all_nine_project_metrics_independently_reproduced':True,
            'result_hashes_match_completion':True,'mean':actual}
    (path/'independent_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path);audit(p.parse_args().path)
