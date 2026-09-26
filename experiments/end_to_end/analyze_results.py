"""Query-level evaluation of frozen, variable-rank candidate selections."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from metrics_core import per_query, summarize

STRATA = ['top', 'middle', 'tail90', 'last']


def read(p):
    return json.loads(p.read_text())


def main(root, interim=False):
    queries = read(root/'queries.json')
    inventory = read(root/'inventory.json')
    models = {c['token']: c['id'] for c in inventory['llms']}
    rows, executions, missing = [], [], []
    frozen = {q['sample_id']: q for q in read(root/'selected_configurations_manifest.json')['queries']}
    for q in queries:
        sid = q['sample_id']
        path = root/'recommendations'/f'{sid}.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen[sid]['recommendation_sha256']
        rec = read(path)
        n = rec['pool_size']
        assert rec['selected_ranks'] == [1, math.ceil(n*.5), math.ceil(n*.9), n]
        scores, ranks, passes, observed_verifiers, aborted, native, simulated = [], [], [], [], [], 0, 0
        for label, config in zip(STRATA, rec['selected']):
            rank = config['recommendation_rank']
            trial = root/'trials'/f'{sid}__r{rank:02d}'
            if not (trial/'execution.json').exists() or not (trial/'judgment.json').exists():
                missing.append(trial.name)
                continue
            e, j = read(trial/'execution.json'), read(trial/'judgment.json')
            assert e['configuration'] == config and config['selection_stratum'] == label
            assert e['query'] == q['query'] and e['backbone'] == models[config['llm']]
            assert j['judge_model'] == 'gpt-5.6-terra'
            scores.append(j['total']); ranks.append(rank)
            native += e['native_tool_calls']; simulated += e['simulated_tool_calls']
            verifier_observed = e.get('verifier') is not None
            verifier_conclusive = verifier_observed and e['verifier']['exit_code'] in [0, 1]
            passed = e['verifier']['exit_code'] == 0 if verifier_conclusive else None
            passes.append(passed)
            observed_verifiers.append(verifier_observed)
            aborted.append(e['termination']=='harness_abort_invalid_tool_arguments')
            api = [read(p)['response'] for p in sorted((trial/'api').glob('[0-9][0-9].json'))]
            executions.append({'sample_id':sid,'dataset':q['dataset'],'task_id':q['task_id'],
                'stratum':label,'rank':rank,'pool_size':n,'rank_percentile':config['rank_percentile'],
                'score':j['total'],'completion':j['completion'],'correctness':j['correctness'],
                'constraints':j['constraints'],'backbone':e['backbone'],'tool_count':len(config['tools']),
                'native_calls':e['native_tool_calls'],'simulated_calls':e['simulated_tool_calls'],
                'verifier_pass':passed,'verifier_observed':verifier_observed,
                'verifier_conclusive':verifier_conclusive,
                'verifier_exit_code':e['verifier']['exit_code'] if verifier_observed else None,
                'termination':e['termination'],'elapsed_sec':e['elapsed_sec'],
                'last_finish_reason':api[-1]['choices'][0].get('finish_reason') if api else None,
                'agent_api_calls':len(api),
                'agent_reported_total_tokens':sum((a.get('usage') or {}).get('total_tokens',0) for a in api)})
        if len(scores) == 4:
            rows.append({'sample_id':sid,'dataset':q['dataset'],'task_id':q['task_id'],'pool_size':n,
                         'ranks':ranks,'strata':STRATA,'scores':scores,'metrics':per_query(scores),
                         'verifier_pass':passes,'verifier_observed':observed_verifiers,
                         'harness_aborted_by_position':aborted,'harness_affected_query':any(aborted),
                         'native_calls':native,'simulated_calls':simulated})
    prefix = 'INTERIM_' if interim else ''
    audit = {'expected_queries':30,'expected_executions':120,'complete_queries':len(rows),
             'complete_executions':len(executions),'missing':missing,'interim':interim}
    (root/(prefix+'completion_audit.json')).write_text(json.dumps(audit,indent=2))
    if not interim:
        assert not missing and len(rows)==30 and len(executions)==120, 'Refusing incomplete final statistics'
    if not rows:
        print(json.dumps(audit)); return
    groups = {'overall':rows}
    for ds in sorted({r['dataset'] for r in rows}):
        groups[ds] = [r for r in rows if r['dataset']==ds]
    metrics = {k:summarize(v) for k,v in groups.items()}
    if 'SkillsBench' in groups:
        g = groups['SkillsBench']
        metrics['SkillsBench']['verifier_pass_by_stratum'] = {
            label:sum(r['verifier_pass'][i] is True for r in g)/len(g) for i,label in enumerate(STRATA)}
        metrics['SkillsBench']['verifier_unobserved_by_stratum'] = {
            label:sum(not r['verifier_observed'][i] for r in g) for i,label in enumerate(STRATA)}
        metrics['SkillsBench']['verifier_inconclusive_by_stratum'] = {
            label:sum(r['verifier_observed'][i] and r['verifier_pass'][i] is None for r in g)
            for i,label in enumerate(STRATA)}
    unaffected=[r for r in rows if not r['harness_affected_query']]
    excluded=[r['sample_id'] for r in rows if r['harness_affected_query']]
    sensitivity={'label':'Post hoc sensitivity prompted by an observed harness validation defect; primary data retained.',
        'exclusion_trigger':'A saved run aborted on a missing command argument before artifact export, independently of its judge score.',
        'excluded_queries':excluded,
        'groups':{k:summarize([r for r in unaffected if k=='overall' or r['dataset']==k])
                  for k in ['overall','AgentSelect','SkillsBench'] if any(k=='overall' or r['dataset']==k for r in unaffected)}}
    (root/(prefix+'sensitivity_excluding_harness_affected_queries.json')).write_text(json.dumps(sensitivity,indent=2))
    for name,data in [('metrics.json',metrics),('per_query_results.json',rows)]:
        (root/(prefix+name)).write_text(json.dumps(data,ensure_ascii=False,indent=2))
    if executions:
        with (root/(prefix+'execution_scores.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(executions[0]));writer.writeheader();writer.writerows(executions)
    print(json.dumps({'completion':audit,'summary':{k:{'n_queries':v['n_queries'],
        'spearman':v['spearman'],'pairwise_concordance':v['pairwise_concordance']} for k,v in metrics.items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path(__file__).resolve().parent)
    p.add_argument('--interim',action='store_true');a=p.parse_args();main(a.root,a.interim)
