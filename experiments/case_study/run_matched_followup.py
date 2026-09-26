"""Prepare/run one isolated four-arm diagnostic from privately supplied records.

Preparation does not execute agents. Running a named arm contacts the separately
configured backbone, simulator and judge endpoints. The original experiment is
read only and the matched follow-up is excluded from its aggregate.
"""
import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path
from agentcomposition.paths import ROOT

S = '<<Whois Lookup_v3&&Check Similarity>>'
D = '<<Whois Lookup_v3&&DNS Lookup>>'
N = '<<Whois Lookup_v3&&NS Lookup>>'
MODEL = '<LLM_deepseek_deepseek-v4-pro>'
ARMS = {'S': {S}, 'SD': {S,D}, 'SN': {S,N}, 'SDN': {S,D,N}}

def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def prepare(source, output):
    if output.exists():raise FileExistsError('Choose a fresh follow-up directory.')
    record_path=source/'recommendations/question_5656.json'
    record=json.loads(record_path.read_text())
    queries=json.loads((source/'queries.json').read_text())
    query=next(q for q in queries if q['sample_id']=='question_5656')
    chosen={}
    for label, tools in ARMS.items():
        matches=[r for r in record['ranked_pool'] if r['llm']==MODEL and set(r['tools'])==tools]
        if len(matches)!=1:raise ValueError('Expected one saved configuration per fixed arm.')
        row=dict(matches[0]);row['recommendation_rank']=row['stage_rank'];chosen[label]=row
    output.mkdir(parents=True)
    for label,row in chosen.items():
        arm=output/label;arm.mkdir()
        shutil.copy2(source/'inventory.json',arm/'inventory.json')
        dump(arm/'queries.json',[query])
        dump(arm/'recommendations/question_5656.json',{**record,'selected':[row]})
        dump(arm/'arm_config.json',{'label':label,'saved_candidate':row})
        for name in ['tool_cache','simulator_cache','simulator_state','native_cache']:
            if (source/name).is_dir():shutil.copytree(source/name,arm/name)
            else:(arm/name).mkdir()
    dump(output/'manifest.json',{
        'purpose':'Post-hoc matched four-arm comparison; excluded from original aggregate',
        'source_record_sha256':hashlib.sha256(record_path.read_bytes()).hexdigest(),
        'cache_policy':'Each arm starts from its own identical copy of the original caches and simulator state',
        'arms':{label:{'rank':row['stage_rank'],'tools':row['tools']} for label,row in chosen.items()},
        'repetitions_per_arm':1,
    })

def run(output,label):
    arm=output/label
    if not (arm/'arm_config.json').exists():raise FileNotFoundError('Prepare the follow-up first.')
    previous=os.environ.get('E2E_ROOT');os.environ['E2E_ROOT']=str(arm)
    try:
        harness=runpy.run_path(str(ROOT/'experiments/end_to_end/run_experiment.py'),run_name='matched_case_harness')
        query=json.loads((arm/'queries.json').read_text())[0]
        row=json.loads((arm/'arm_config.json').read_text())['saved_candidate']
        execution=harness['execute'](query,row)
        judgment=harness['judge'](execution)
        dump(arm/'summary.json',{'arm':label,'rank':row['stage_rank'],'raw_critic':row['critic_raw'],
            'final_rerank':row['search_score'],'judge_total':judgment['total'],
            'termination':execution['termination'],'tool_calls':execution['tool_calls'],
            'native_tool_calls':execution['native_tool_calls'],'simulated_tool_calls':execution['simulated_tool_calls']})
    finally:
        if previous is None:os.environ.pop('E2E_ROOT',None)
        else:os.environ['E2E_ROOT']=previous

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/end_to_end')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/case_study/followup')
    parser.add_argument('--arm',choices=list(ARMS))
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.source,args.output)
    else:
        if not args.arm:parser.error('--arm is required for execution')
        run(args.output,args.arm)
if __name__=='__main__':main()
