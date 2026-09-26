#!/usr/bin/env python3
"""Check complete six-group result files and actual search/selection boundaries."""
import argparse,hashlib,json,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root
order=['Free','Greedy','Beam','Critic','Beam+Critic','Critic+Cartesian']
sha=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
readlines=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
key=lambda p:(p.get('llm',p.get('llm_token')),tuple(sorted(set(p.get('tools',p.get('tool_tokens',[])))-{'<TOOL_EMPTY>'})))
allrows={v:readlines(root/v/'results.jsonl') for v in order}
manifest=readlines(root/'Free/sample_manifest.jsonl');assert len(manifest)==100
report={'passed':False,'n':100,'variants':{},'search_critic_event_count':0,'gold_used_only_for_evaluation':True}
for v,rows in allrows.items():
    config=json.loads((root/v/'config.json').read_text())
    assert len(rows)==100
    assert readlines(root/v/'sample_manifest.jsonl')==manifest
    zero=[]
    for sample,r in zip(manifest,rows):
        assert r['ok'] and r['dataset_example']==sample and r['config_sha256']==sha(config)
        preds=r['results'];assert 1<=len(preds)<=10
        assert len({key(p) for p in preds})==len(preds)
        assert [p['rank'] for p in preds]==list(range(1,len(preds)+1))
        if v in ('Free','Greedy'):
            assert len(preds)==1
            assert not r['generation'].get('critic_api_events')
            assert r['config']['controlled']==int(v=='Greedy')
        if v=='Beam+Critic':
            trace=r['generation']['search_trace'];assert trace['search_critic_enabled'] is True
            events=r['generation']['critic_api_events']
            assert not any(e.get('error') for e in events)
            assert events[-1]['stage']=='final_rerank'
            calls=[e for e in events if e['stage']!='final_rerank']
            assert all(e['stage'].startswith('tool_depth_') and int(e['stage'].split('_')[-1])>=2 for e in calls)
            report['search_critic_event_count']+=len(calls)
            if not calls:zero.append(sample['sample_id'])
            for lev in trace['levels']:
                if lev.get('prune'):
                    assert lev['prune']['retained_count']<=2
                for n in lev.get('nodes',[]):
                    real=[t for t in n['tools'] if t!='<TOOL_EMPTY>']
                    if len(real)<2:assert n['critic_raw'] is None
                    elif lev['stage']!='llm_roots':assert n['critic_raw'] is not None and math.isfinite(n['critic_raw'])
            pool=trace['final_rerank_pool_before_cap']
            assert all(n['search_score']==n['critic_raw'] for n in pool)
            assert [key(p) for p in preds]==[key(p) for p in pool[:10]]
            assert all(pool[i]['critic_raw']>=pool[i+1]['critic_raw'] for i in range(len(pool)-1))
    report['variants'][v]={'samples':len(rows),'candidate_min':min(len(r['results']) for r in rows),'candidate_max':max(len(r['results']) for r in rows),'search_critic_not_reached_samples':zero}
for beam,critic,cart in zip(allrows['Beam'],allrows['Critic'],allrows['Critic+Cartesian']):
    sid=beam['dataset_example']['sample_id']
    pool=json.loads((root/'shared_completed_pools'/(sid+'.json')).read_text())['nodes']
    expected_beam=sorted(pool,key=lambda n:(-n['generator_only_score'],-n['generator_avg_logprob'],n['node_id']))[:10]
    expected_critic=sorted(pool,key=lambda n:(-n['critic_raw_v3'],key(n)))[:10]
    assert [key(p) for p in beam['results']]==[key(p) for p in expected_beam]
    assert [key(p) for p in critic['results']]==[key(p) for p in expected_critic]
    toolsets={key(p)[1] for p in critic['results']}
    assert all(key(p)[1] in toolsets for p in cart['results'])
    assert cart['available_pool_count']==len(toolsets)*10
    scores=[p['score_details']['critic_raw'] for p in cart['results']]
    assert scores==sorted(scores,reverse=True)
summary=json.loads((root/'SUMMARY.json').read_text());assert summary['complete'] and not summary['missing']
for v in order:
    metrics=summary['variants'][v]
    assert metrics['n']==100 and metrics['official_evaluator_max_difference']<1e-12
    assert metrics['raw']['SUCR@10']+1e-12>=metrics['raw']['OGR@10']
    assert metrics['raw']['Oracle-RDCR@10']+1e-12>=metrics['raw']['RDCR@10']
replay=json.loads((root/'runtime_replay_check.json').read_text())
assert replay['n']==3 and replay['pool_compatible']
assert all(c['max_generator_avg_difference']==0 for c in replay['checks'])
report.update(passed=True,official_evaluator_max_difference=max(r['official_evaluator_max_difference'] for r in summary['variants'].values()),runtime_replay_n=3,runtime_replay_max_difference=0.)
(root/'FINAL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
