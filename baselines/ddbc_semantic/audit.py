#!/usr/bin/env python3
"""Independent target-only metrics, component legality, and provenance audit."""
import argparse,collections,hashlib,json,math,re
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args();out=a.results
    cat=json.loads((a.data/'catalog.json').read_text());mapping={c['id']:c for c in cat}
    records=[json.loads(x) for x in (out/'results.jsonl').open()];blind=[json.loads(x) for x in (out/'predictions_blind.jsonl').open()]
    assert len(records)==len(blind)==100
    assert len({r['dataset_example']['sample_id'] for r in records})==100
    agg=collections.defaultdict(list)
    for r,b in zip(records,blind):
        assert b['sample_id']==r['dataset_example']['sample_id'];assert b['results']==r['results']
        assert 1<=len(r['results'])<=10
        configs=set();scores=[]
        for v in b['all_ranked_proposals']:
            ids=v['components'];assert 1<=len(ids)<=7;assert mapping[ids[0]]['kind']=='llm'
            assert all(mapping[i]['kind']=='tool' for i in ids[1:]);assert len(ids)==len(set(ids))
            assert abs(v['ranking_score']-v['pseudo_logp_per_code']-v['length_logp'])<1e-10
        for rank,v in enumerate(r['results'],1):
            ids=v['components'];assert v['rank']==rank
            assert v['llm_token']==mapping[ids[0]]['token'];assert v['tool_tokens']==[mapping[i]['token'] for i in ids[1:]]
            key=(v['llm_token'],tuple(sorted(v['tool_tokens'])));assert key not in configs;configs.add(key);scores.append(v['ranking_score'])
        assert scores==sorted(scores,reverse=True)
        target=r['dataset_example']['target'].split('<SPECIAL_END>')[0].split('Explanation:')[0]
        llm=re.findall(r'<LLM_[^<>\n\r]+>',target)[0]
        gt=set(re.findall(r'<<[^<>\n\r]+>>|<TOOL_[^<>\n\r]+>',target))-{'<TOOL_SEP>','<TOOL_EMPTY>'};gc=gt|{llm}
        cr=[];th=[];hits=[]
        for i,v in enumerate(r['results']):
            pt=set(v['tool_tokens']);n=len(pt&gt);rec=n/len(gt) if gt else 1.;precision=n/len(pt) if pt else float(not gt)
            cr.append(len((pt|{v['llm_token']})&gc)/len(gc));th.append(gt<=pt);hits.append(llm==v['llm_token'] and gt<=pt)
            if i==0:
                agg['top1_tool_recall'].append(rec);agg['top1_tool_precision'].append(precision);agg['top1_tool_f1'].append(2*precision*rec/(precision+rec) if precision+rec else 0.)
        agg['top1_component_recall'].append(cr[0]);agg['tool_hit@10'].append(float(any(th)));agg['cr_hit@1'].append(float(hits[0]));agg['cr_hit@10'].append(float(any(hits)))
        agg['cr_mrr@10'].append(next((1/(i+1) for i,v in enumerate(hits) if v),0.))
        weights=[1/math.log2(i+2) for i in range(10)];agg['rdcr@10'].append(sum(v*w for v,w in zip(cr,weights))/sum(weights))
    mean={k:sum(v)/len(v) for k,v in agg.items()};official=json.loads((out/'evaluation/top10_metrics.json').read_text())['mean']
    assert all(abs(v-official[k])<1e-12 for k,v in mean.items())
    done=json.loads((out/'completed.json').read_text());assert sha(out/'results.jsonl')==done['results_sha256'];assert sha(out/'predictions_blind.jsonl')==done['predictions_sha256']
    report={'passed':True,'n':100,'all_outputs_are_legal_component_combinations':True,'all_rankings_match_declared_score':True,'nine_metrics_independently_reproduced':True,'file_hashes_match':True,'candidate_counts':dict(collections.Counter(len(r['results']) for r in records)),'mean':mean}
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
