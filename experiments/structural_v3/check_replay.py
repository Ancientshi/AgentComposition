from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,sys
from run_variants import ROOT,EXP,base,write
old=ROOT/'outputs/table1_ours_v10_compact_fixedtest100/per_sample'
new=EXP/'BeamReplay_smoke/per_sample'
checks=[]
for p in sorted(new.glob('*.json')):
    a=json.loads((old/p.name).read_text());b=json.loads(p.read_text())
    assert b.get('ok'),p
    ta=a['generation']['search_trace'];tb=b['generation']['search_trace']
    def index(trace):
        return {(n['llm'],tuple(sorted(n['tools']))):n for n in trace['final_rerank_pool_before_cap']}
    ia,ib=index(ta),index(tb)
    same=set(ia)==set(ib)
    diff=max((abs(ia[k]['generator_avg_logprob']-ib[k]['generator_avg_logprob']) for k in ia.keys()&ib.keys()),default=0)
    stages=[]
    for x,y in zip(ta['levels'],tb['levels']):
        def retained(z):
            return z['prune']
        # Full prune summaries may include float scores; inspect selected node identities.
        stages.append({'stage':x['stage'],'prune_old':retained(x),'prune_new':retained(y)})
    checks.append({'sample_id':a['dataset_example']['sample_id'],'same_complete_pool':same,
                   'old_pool_count':len(ia),'new_pool_count':len(ib),'max_generator_avg_difference':diff,
                   'same_generated_node_count':ta['generated_node_count']==tb['generated_node_count'],
                   'stage_comparison':stages})
report={'n':len(checks),'pool_compatible':bool(checks) and all(c['same_complete_pool'] and c['same_generated_node_count'] for c in checks),'checks':checks}
write(EXP/'runtime_replay_check.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
for c in checks:print({k:v for k,v in c.items() if k!='stage_comparison'})
if not report['pool_compatible']:sys.exit(1)
