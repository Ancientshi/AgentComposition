"""Preserve an observed invalid-call abort; never replay or regenerate its actions.

The frozen executor indexed arguments['command'] without validating its presence.
It removed the container in finally, so artifact presence at the abort is unknown.
This postprocessor records that limitation and submits only saved evidence to the
unchanged judge. The original responses and exception remain untouched.
"""
import hashlib
import json
from pathlib import Path
import time

import run_experiment as runner

ROOT=runner.ROOT


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    queries={q['sample_id']:q for q in read(ROOT/'queries.json')}
    frozen=read(ROOT/'pre_execution_hashes.json')
    assert sha(ROOT/'run_experiment.py')==frozen['run_experiment.py']
    for error_path in sorted((ROOT/'trials').glob('*/infrastructure_error.json')):
        trial=error_path.parent;error=read(error_path)
        if error.get('type')!='KeyError' or error.get('error')!="'command'":continue
        if (trial/'execution.json').exists():
            if (trial/'abort_resolution.json').exists():
                print(json.dumps({'trial':trial.name,'judgment':runner.judge(read(trial/'execution.json'))['total']}),flush=True)
            continue
        start=read(trial/'started.json');q=queries[start['sample_id']]
        assert q['dataset']=='SkillsBench'
        responses=sorted((trial/'api').glob('[0-9][0-9].json'))
        last=responses[-1];raw=read(last)['response'];message=raw['choices'][0]['message']
        calls=message.get('tool_calls',[])
        assert len(calls)==1 and calls[0]['function']['name']=='execute'
        arguments=json.loads(calls[0]['function']['arguments'])
        assert isinstance(arguments,dict) and 'command' not in arguments
        partial=read(trial/'trajectory_partial.json')
        assert partial['messages']==read(last.with_suffix('.request.json'))['messages']
        assert int(last.stem)==len(responses)-1
        note=('The agent emitted execute without the required command argument. '
              'The frozen harness raised KeyError before executing this call or returning a tool error, '
              'then removed the container before artifact export and independent verification. '
              'File presence at the abort is unknown. No action was replayed and no new backbone response was requested.')
        resolution={'classification':'invalid_agent_call_and_harness_validation_abort',
                    'documented_before_abort_judgment_unix':time.time(),
                    'original_exception':error,'last_response_file':str(last.relative_to(trial)),
                    'source_sha256':{str(p.relative_to(trial)):sha(p) for p in [error_path,last,last.with_suffix('.request.json'),trial/'trajectory_partial.json',trial/'started.json']},
                    'note':note,'backbone_rerun':False,'artifact_presence_at_abort':'unknown',
                    'sensitivity_policy':'Additionally recompute query-level metrics after excluding every query containing this kind of harness-aborted trial; retain all trials in the primary end-to-end summary.'}
        runner.dump(trial/'abort_resolution.json',resolution)
        events=list(partial['events'])
        events.append({'turn':int(last.stem),'tool_name':'execute','arguments':arguments,
                       'output':{'mode':'validation','error':note,'reconstructed_from_saved_response_and_exception':True,'returned_to_agent':False}})
        artifacts=[{'requested_path':name,'exists':False,'sha256':None,'text':None,'chars':0,
                    'availability':'not_exported_due_to_harness_abort','on_disk_exists_at_abort':None,
                    'evidence_limit':note} for name in runner.ARTIFACTS[q['task_id']]]
        record={'sample_id':q['sample_id'],'dataset':q['dataset'],'task_id':q['task_id'],
                'recommendation_rank':start['rank'],'backbone':start['model'],'configuration':start['config'],
                'query':q['query'],'messages':partial['messages']+[{k:v for k,v in message.items() if k in ['role','content','tool_calls']}],
                'events':events,'final_answer':'','termination':'harness_abort_invalid_tool_arguments',
                'artifacts':artifacts,'verifier':None,'elapsed_sec':error['time']-start['started_at'],
                'tool_calls':len(events),
                'native_tool_calls':sum(e['output'].get('mode') in ['native_execution','native_counterpart'] for e in events),
                'simulated_tool_calls':sum(e['output'].get('mode')=='llm_simulator' for e in events),
                'record_finalized_from_saved_aborted_attempt':True,'harness_incident':note}
        runner.dump(trial/'execution.json',record)
        judgment=runner.judge(record)
        print(json.dumps({'trial':trial.name,'score':judgment['total'],'backbone_rerun':False}),flush=True)


if __name__=='__main__':main()
