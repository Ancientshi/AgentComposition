#!/usr/bin/env python3
"""Frozen-test V10 structural ablation; all changes isolated from original runners."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse, contextlib, copy, hashlib, inspect, json, os, random, sys, time
from pathlib import Path
ROOT = AC_ROOT
sys.path.insert(0, str(ROOT/'training/generator'))
import run_infer_table1_compact as base
from compact_context import build_prompt
VARIANTS = ['Free','Greedy','Beam','Critic','Beam+Critic','Critic+Cartesian']
EXP = ROOT/'outputs/structural_v10_criticv4_beam1-2_fixed100'
MODEL = ROOT/'checkpoints/generative_v10_compact'
BASE_MODEL = Path(str(AC_BASE_MODEL))
CRITIC = AC_ENV('CRITIC_URL','http://127.0.0.1:8015')
WEIGHTS = 'becd7447dcac496c1216d55998cf399c2bb9efcc6a9f49a75dd5189e447bc061'
MANIFEST_SHA = '4dc31d07fea9854d7e826f943f187d960e4f6a02dda2f4f194d5e7c2dd49bd5a'
write = base.write_json

def replace_once(src, old, new):
    if src.count(old) != 1: raise RuntimeError('Code anchor changed: '+old[:90])
    return src.replace(old,new,1)

def install(sota, final_mode):
    # Keep the Table 1 search unchanged except explicit partial-critic insertion.
    src = inspect.getsource(sota.critic_guided_tree_search)
    old = '''        critic_event = {
            "stage": stage,
            "skipped": True,
            "reason": "Search-time Bundle Critic disabled in SOTA; generator-only beam pruning",
            "candidate_count": len(expansions),
        }
        assign_generator_only_scores(expansions, length_penalty=length_penalty)'''
    new = '''        critic_eligible = [n for n in expansions if len([t for t in n.tools if t != tool_empty_token]) >= 2]
        critic_ineligible = [n for n in expansions if n not in critic_eligible]
        assign_generator_only_scores(critic_ineligible, length_penalty=length_penalty)
        critic_event = {"stage": stage, "skipped": True, "candidate_count": len(expansions), "reason": "fewer than two real tools"}
        if critic_eligible:
            critic_event = critic.score_nodes(query=query, nodes=critic_eligible, evidence_context=context, stage=stage)
            assign_search_scores(critic_eligible, mode=search_score_mode, critic_weight=critic_weight,
                                 generator_weight=generator_weight, length_penalty=length_penalty)'''
    src = replace_once(src,old,new)
    src = src.replace('"search_critic_enabled": False','"search_critic_enabled": True')
    src = src.replace('"critic_policy": "final_rerank_only"','"critic_policy": "delayed_min2_search_and_final"')
    src = src.replace('"search_critic=False generator_only=True"','"search_critic=True delayed_min2=True"')
    old_final = '''    assign_search_scores(
        completed,
        mode=search_score_mode,
        critic_weight=critic_weight,
        generator_weight=generator_weight,
        length_penalty=length_penalty,
    )'''
    if final_mode == 'raw':
        src = replace_once(src,old_final,'''    for node in completed:
        node.search_score = float(node.critic_raw)
        node.score_components = {"mode": "critic_raw_only", "critic_raw": node.critic_raw}''')
    exec(compile(src, sota.__file__, 'exec'),sota.__dict__)
    # Remove only final-only guard for the explicit search ablation.
    src = inspect.getsource(sota.BundleCriticClient.score_nodes)
    import textwrap
    src = textwrap.dedent(src)
    start = src.index('    if stage != "final_rerank":')
    end = src.index('    missing: List[SearchNode]',start)
    src = src[:start]+src[end:]
    ns = dict(sota.__dict__)
    exec(compile(src,sota.__file__,'exec'),ns)
    sota.BundleCriticClient.score_nodes = ns['score_nodes']

def load_inputs():
    manifest_path=ROOT/'datasets/generative_v10_compact/test_manifest.frozen.jsonl'
    assert base.file_sha(manifest_path)==MANIFEST_SHA
    manifest, rows = base.source_records(ROOT/'outputs/baseline4_rag_llama_instruct_seed42_n100')
    assert manifest == base.read_lines(manifest_path) and len(manifest)==100
    provenance=json.loads((MODEL/'training_provenance.json').read_text())
    assert provenance['context_sha256']==base.file_sha(ROOT/'training/generator/compact_context.py')
    assert provenance['max_prompt_tokens']==6144
    assert base.file_sha(MODEL/'adapter_model.safetensors')=='97da97b8d7dde0a21792df9968112fd972da69677fc2324cae1a5af347047e37'
    return manifest,rows

def run(variant, final_mode, limit=0, check=False, shard=0, shards=1):
    manifest,rows=load_inputs()
    health=base.critic_health(CRITIC)
    assert health['model']['weights_sha256']==WEIGHTS
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(BASE_MODEL),local_files_only=True)
    sota=base.load_module(ROOT/'inference/run_infer_v13_batch_eval_sota.py','ablation_sota')
    sota.build_prompt_v12=lambda *,context,query,**kw:build_prompt(context,query,tokenizer,6144)[0]
    if variant=='Beam+Critic': install(sota,final_mode)
    sota.BundleCriticClient.analyze_node=lambda self,**kw:{'skipped':True,'reason':'ranking only'}
    settings=copy.deepcopy(base.SETTINGS)
    settings.update(model_dir=str(MODEL),base_model_name=str(BASE_MODEL),critic_url=CRITIC,
                    max_new_tokens=64,top_k=1,num_beams=1,search_checkpoint_interval=0)
    if variant in ('Free','Greedy'):
        settings.update(controlled=int(variant=='Greedy'),search_mode='greedy',
                        allow_free_fallback=int(variant=='Free'),critic_required=0)
    argv=sys.argv
    try:
        sys.argv=[sota.__file__]+[t for k,v in settings.items() for t in ('--'+k,str(v))]
        args=sota.parse_args()
    finally: sys.argv=argv
    if check:
        print('[CHECK]',variant,'passed; model not loaded',flush=True);return
    if limit:manifest,rows=manifest[:limit],rows[:limit]
    manifest,rows=manifest[shard::shards],rows[shard::shards]
    out=EXP/(variant + ('_smoke' if limit else '') + (f'_shard{shard}' if shards > 1 else ''))
    config={'variant':variant,'settings':settings,'final_ranking':final_mode,'critic_model':health['model'],
            'manifest_sha256':MANIFEST_SHA,'runner_sha256':base.file_sha(__file__),
            'sota_sha256':base.file_sha(sota.__file__),'n':len(manifest),'seed':42,'shard':shard,'shards':shards,
            'model_sha256':base.file_sha(MODEL/'adapter_model.safetensors'),
            'compact_sha256':base.file_sha(ROOT/'training/generator/compact_context.py'),
            'runtime_versions':{n:__import__('importlib.metadata',fromlist=['version']).version(n) for n in ['torch','transformers','peft','numpy']}}
    cp=out/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'Resume config mismatch'
    write(cp,config)
    (out/'sample_manifest.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in manifest))
    fingerprint=base.sha(config)
    import torch,numpy as np
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    # Cap our allocator to leave the other task's existing allocation intact.
    torch.cuda.set_per_process_memory_fraction(0.43,0)
    completed=[]
    for idx,(sample,row) in enumerate(zip(manifest,rows),1):
        path=out/'per_sample'/(sample['sample_id']+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get('ok'):
                assert saved['config_sha256']==fingerprint and saved['dataset_example']==sample
                completed.append(saved);continue
        a=copy.copy(args);a.query=row['query'];a.context=row['context']['text'];a.output_json=str(path)
        log=out/'logs'/(sample['sample_id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        started=time.time()
        try:
            with log.open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
                record=sota.run_pipeline(a)
            for rank,r in enumerate(record['results'],1):r['rank']=rank
            if variant in ('Free','Greedy'):
                assert len(record['results'])==1
                assert not record['generation'].get('critic_api_events')
            else:
                events=record['generation']['critic_api_events']
                assert any(e['stage']=='final_rerank' for e in events)
                assert not any(e.get('error') for e in events)
                record['generation'].update(search_critic_enabled=variant=='Beam+Critic',final_critic_reranking=True,
                                            sota_policy='structural_search_and_final_critic')
                record['config'].update(search_critic_enabled=variant=='Beam+Critic')
            record.update(ok=True,dataset_example=sample,config_sha256=fingerprint,variant=variant,
                          context=row['context'],retrieval=row['retrieval'],latency_sec=time.time()-started)
            write(path,record);completed.append(record)
            write(out/'status.json',{'state':'running','completed':idx,'total':len(rows)})
            print('[OK]',variant,idx,len(rows),round(time.time()-started,1),flush=True)
        except BaseException as exc:
            write(out/'status.json',{'state':'failed','completed':len(completed),'error':repr(exc),'sample':sample['sample_id'],'log':str(log)})
            raise
    (out/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in completed))
    write(out/'status.json',{'state':'complete','completed':len(completed),'total':len(rows)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['Free','Greedy','Beam+Critic','BeamReplay'],required=True)
    ap.add_argument('--final_mode',choices=['raw','hybrid'],default='raw');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--check',action='store_true');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);a=ap.parse_args()
    # Serialize duplicate invocations of the same variant; resume after the owner finishes.
    import fcntl
    EXP.mkdir(parents=True,exist_ok=True)
    lock_name=a.variant+('_smoke' if a.limit else '')+f'_shard{a.shard}.lock'
    with (EXP/lock_name).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run(a.variant,a.final_mode,a.limit,a.check,a.shard,a.shards)
