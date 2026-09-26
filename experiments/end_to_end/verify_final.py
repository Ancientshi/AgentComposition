"""Verify frozen selection, execution identities, tool access, artifacts and blind scores."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(x):return sha(json.dumps(x,sort_keys=True,ensure_ascii=False).encode())


def main(interim=False):
    checks=collections.Counter();modes=collections.Counter();finish=collections.Counter();models=collections.Counter()
    diagnostics=[];prompts=set();sim_prompts=set();seen=set();usage=collections.Counter();verifier_errors=[]
    for manifest in ['pre_generation_hashes.json','pre_execution_hashes.json']:
        for name,expected in read(ROOT/manifest).items():
            assert sha((ROOT/name).read_bytes())==expected,(manifest,name)
            checks['frozen_files_unchanged']+=1
    inventory=read(ROOT/'inventory.json')
    components={c['token']:c for c in inventory['components']}
    llms={m['token']:m['id'] for m in inventory['llms']}
    freeze=read(ROOT/'selected_configurations_manifest.json')
    frozen={r['sample_id']:r for r in freeze['queries']}
    for q in read(ROOT/'queries.json'):
        sid=q['sample_id'];rec_path=ROOT/'recommendations'/f'{sid}.json';r=read(rec_path)
        assert sha(rec_path.read_bytes())==frozen[sid]['recommendation_sha256']
        assert r['query_sha256']==sha(q['query'].encode())==q['query_sha256']
        assert r['context_sha256']==sha((ROOT/'context.txt').read_bytes())
        src='agentselect_expanded.py' if q['dataset']=='AgentSelect' else 'skillsbench_expanded.py'
        assert r['source_sha256']==sha((ROOT/'source_snapshot'/src).read_bytes())
        n=len(r['ranked_pool']);positions=[1,math.ceil(n*.5),math.ceil(n*.9),n]
        assert n>=100 and positions==r['selected_ranks']
        order='search_score' if q['dataset']=='AgentSelect' else 'generator_avg_logprob'
        values=[c[order] for c in r['ranked_pool']]
        assert all(a>=b for a,b in zip(values,values[1:]))
        assert len({(c['llm'],tuple(sorted(c['tools']))) for c in r['ranked_pool']})==n
        generation=read(ROOT/'generation'/f'{sid}.json')
        trace=generation['generation']['search_trace']
        assert trace['search_critic_enabled'] is False
        assert bool(trace['final_critic_reranking'])==(q['dataset']=='AgentSelect')
        pruned_ids={node for level in trace['levels'] for node in level.get('prune',{}).get('pruned_node_ids',[])}
        recovered_ids={node for stage in trace['pruned_prefix_completion'] for node in stage['node_ids']}
        for candidate in r['ranked_pool']:
            assert candidate['is_complete'] and candidate['llm'] in llms
            assert len(candidate['tools'])==len(set(candidate['tools']))
            assert len(candidate['tools'])<=(6 if q['dataset']=='AgentSelect' else 10)
            assert all(t in components for t in candidate['tools'])
            assert abs(candidate['generator_avg_logprob']-candidate['generator_logprob']/max(1,candidate['generator_token_count']))<1e-12
            if candidate['termination_reason']=='pruned_prefix_then_scored_end':
                assert candidate['node_id'] in recovered_ids and candidate['parent_id'] in pruned_ids
                checks['completed_candidates_traced_to_actual_pruning']+=1
        checks['ranked_unique_pools_verified']+=1
        for label,rank,config in zip(['top','middle','tail90','last'],positions,r['selected']):
            clean=dict(config);assert clean.pop('selection_stratum')==label
            assert clean==r['ranked_pool'][rank-1]
            trial=ROOT/'trials'/f'{sid}__r{rank:02d}'
            if interim and not all((trial/name).exists() for name in ['execution.json','judgment.json']):
                continue
            e=read(trial/'execution.json');j=read(trial/'judgment.json');start=read(trial/'started.json')
            assert start['started_at']>=freeze['frozen_before_execution_unix']
            assert e['query']==q['query'] and e['configuration']==config and e['backbone']==llms[config['llm']]
            assert e['messages'][1]=={'role':'user','content':q['query']}
            key=(sid,rank);assert key not in seen;seen.add(key);models[e['backbone']]+=1
            allowed={components[t]['function_name'] for t in config['tools'] if components[t]['kind']=='api'}
            if q['dataset']=='SkillsBench':allowed.add('execute')
            for event in e['events']:
                output=event['output'];mode=output.get('mode','unknown');modes[mode]+=1
                if event['tool_name'] not in allowed:assert mode in ['validation','budget'] and 'error' in output
                if mode=='llm_simulator':assert output['simulator_model']=='gpt-5.6-luna' and output['synthetic'] is True
            reasons=[]
            for p in sorted((trial/'api').glob('[0-9][0-9].json')):
                req=read(p.with_suffix('.request.json'));record=read(p);resp=record['response']
                assert req['model']==resp['model']==e['backbone']
                assert req['temperature']==0 and req['enable_thinking'] is False
                assert req['max_tokens']==(6000 if q['dataset']=='SkillsBench' else 4000)
                assert {t['function']['name'] for t in req.get('tools',[])}==allowed
                assert req['messages'][1]=={'role':'user','content':q['query']}
                assert digest(req)==record['request_sha256']
                reason=resp['choices'][0].get('finish_reason');finish[str(reason)]+=1;reasons.append(reason)
                checks['backbone_responses_verified']+=1
                for k in ['prompt_tokens','completion_tokens','total_tokens']:usage['backbone_'+k]+=(resp.get('usage') or {}).get(k,0)
            if e['termination']=='harness_abort_invalid_tool_arguments':
                resolution=read(trial/'abort_resolution.json')
                assert resolution['backbone_rerun'] is False and resolution['artifact_presence_at_abort']=='unknown'
                for name,expected in resolution['source_sha256'].items():
                    assert sha((trial/name).read_bytes())==expected
                partial=read(trial/'trajectory_partial.json')
                last=read(trial/resolution['last_response_file'])['response']['choices'][0]['message']
                assert len(last['tool_calls'])==1
                call=last['tool_calls'][0];arguments=json.loads(call['function']['arguments'])
                assert call['function']['name']=='execute' and 'command' not in arguments
                assert e['events'][:-1]==partial['events'] and e['events'][-1]['arguments']==arguments
                assert e['events'][-1]['output']['returned_to_agent'] is False
                assert e['verifier'] is None and e['final_answer']==''
                checks['aborted_trials_preserved_without_backbone_rerun']+=1
            for artifact in e['artifacts']:
                if not artifact['exists']:
                    if artifact.get('availability')=='not_exported_due_to_harness_abort':
                        assert artifact['on_disk_exists_at_abort'] is None
                        checks['artifact_state_unobserved_after_harness_abort']+=1
                    else:checks['missing_required_artifacts']+=1
                    continue
                p=trial/'artifacts'/Path(artifact['requested_path']).name
                assert p.is_file() and not p.is_symlink() and sha(p.read_bytes())==artifact['sha256']
                assert p.read_text(errors='replace')[:55000]==artifact['text'];checks['artifacts_verified']+=1
            req=read(trial/'judge_api.request.json');record=read(trial/'judge_api.json');resp=record['response']
            assert req['model']==resp['model']==j['judge_model']=='gpt-5.6-terra'
            assert req['reasoning_effort']=='low' and req['response_format']=={'type':'json_object'}
            assert digest(req)==record['request_sha256'];prompts.add(sha(req['messages'][0]['content'].encode()))
            body=json.loads(req['messages'][1]['content'])
            assert set(body)=={'user_request','final_answer','submitted_artifacts','termination','observable_evidence','independent_verifier'}
            assert body['user_request']==q['query'] and body['final_answer']==e['final_answer']
            assert body['submitted_artifacts']==e['artifacts'] and body['independent_verifier']==e['verifier']
            assert j['blind_payload_sha256']==digest(body)
            assert sum(j[k] for k in ['completion','correctness','constraints'])==j['total']
            for k,maximum in [('completion',4),('correctness',4),('constraints',2)]:assert 0<=j[k]<=maximum and j[k]*2==int(j[k]*2)
            for k in ['prompt_tokens','completion_tokens','total_tokens']:usage['judge_'+k]+=(resp.get('usage') or {}).get(k,0)
            if e.get('verifier') and e['verifier']['exit_code'] not in [0,1]:verifier_errors.append(trial.name)
            retries=[v for p in (trial/'api').glob('*.transport_retries.json') for v in read(p)]
            diagnostics.append({'sample_id':sid,'dataset':q['dataset'],'rank':rank,'stratum':label,
                'score':j['total'],'termination':e['termination'],'last_finish_reason':reasons[-1] if reasons else None,
                'length_responses':reasons.count('length'),'native_calls':e['native_tool_calls'],
                'simulated_calls':e['simulated_tool_calls'],'http_retries':len(retries),
                'missing_artifacts':sum(not a['exists'] and a.get('availability')!='not_exported_due_to_harness_abort' for a in e['artifacts']),
                'unobserved_artifacts':sum(a.get('availability')=='not_exported_due_to_harness_abort' for a in e['artifacts']),
                'verifier_exit_code':e['verifier']['exit_code'] if e.get('verifier') else None})
            checks['blind_judgments_verified']+=1
    for p in (ROOT/'simulator_api').glob('*.json'):
        if p.name.endswith(('.request.json','.transport_retries.json')):continue
        req=read(p.with_suffix('.request.json'));record=read(p);resp=record['response']
        assert req['model']==resp['model']=='gpt-5.6-luna'
        assert req['reasoning_effort']=='low' and digest(req)==record['request_sha256']
        sim_prompts.add(sha(req['messages'][0]['content'].encode()))
        body=json.loads(req['messages'][1]['content'])
        assert set(body)=={'service','endpoint','description','schema','arguments','prior_service_observations'}
        checks['simulator_responses_verified']+=1
        for k in ['prompt_tokens','completion_tokens','total_tokens']:usage['simulator_'+k]+=(resp.get('usage') or {}).get(k,0)
    assert (interim or len(seen)==120) and len(prompts)==(1 if seen else 0) and len(sim_prompts)<=1
    out={'status':'interim_verified' if interim else 'verified','execution_count':len(seen),'query_count':30,'checks':dict(checks),
         'backbone_counts':dict(models),'tool_event_modes':dict(modes),'finish_reasons':dict(finish),
         'primary_judge_system_sha256':next(iter(prompts),None),'simulator_system_sha256':list(sim_prompts),
         'verifier_runtime_errors':verifier_errors,'saved_successful_response_usage':dict(usage),
         'usage_scope':'Saved successful responses, excluding failed requests; not an account bill.',
         'selection_frozen_before_all_executions':True,'no_score_based_exclusions':True}
    prefix='INTERIM_' if interim else ''
    (ROOT/(prefix+'final_verification.json')).write_text(json.dumps(out,indent=2))
    (ROOT/(prefix+'execution_diagnostics.json')).write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--interim',action='store_true')
    main(parser.parse_args().interim)
