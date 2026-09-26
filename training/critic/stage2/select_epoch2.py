"""Execute the user's explicit stop-at-epoch-2 decision, before test scoring."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
from prepare import sha

def main():
    r=Path(__file__).resolve().parent;b=r.parent;out=b/'outputs/bundle_critic_reference_v4';statefile=r/'pipeline_state.json'
    request=json.loads((r/'user_stop_request.json').read_text())
    assert request['selected_epoch']==2 and request['alpha']==1
    if Path(f"/proc/{request['stopped_pid']}").exists():raise RuntimeError('Training process has not exited yet')
    assert not (out/'best_critic.pt').exists() and not (out/'COMPLETE.json').exists()
    ck=torch.load(out/'last_critic.pt',map_location='cpu',weights_only=False)
    assert ck['epoch']==2,'Do not silently use a different epoch'
    del ck
    shutil.copy2(out/'last_critic.pt',out/'best_critic.pt')
    baseline=b/'outputs/bundle_critic_terra_v3/best_critic.pt'
    config={'alpha':1.0,'epoch':2,'baseline_checkpoint':str(baseline),'baseline_sha256':sha(baseline),
            'formula':'Raw V4 epoch 2; alpha=1 means no V3 mixing.',
            'selected_on':'Explicit user decision after epoch-2 validation and before any V4 test scoring.',
            'original_strict_recall_guard_passed':False,'user_accepted_validation_recall_decrease':True}
    (out/'ranking_config.json').write_text(json.dumps(config,indent=2)+'\n')
    val=json.loads((out/'validation_epoch_2.json').read_text())
    (out/'best_metrics.json').write_text(json.dumps({**val,'selection_decision':config},indent=2)+'\n')
    met=json.loads((out/'metrics.json').read_text());met['selected_epoch']=2;met['selection_decision']=config
    (out/'metrics.json').write_text(json.dumps(met,indent=2)+'\n')
    result={'training_complete':True,'epochs_completed':2,'third_epoch_interrupted_at_user_request':True,'selected_epoch':2,
            'accepted_on_synthetic_validation':False,'accepted_by_user':True,'original_strict_recall_guard_passed':False,
            'deployment_validated':False,'new_llm_calls':0,'test_used_for_selection':False}
    (out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    def state(stage,**kw):statefile.write_text(json.dumps({'stage':stage,'time':time.time(),'pid':os.getpid(),**kw},indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    try:
        state('evaluate_user_selected',selection=config)
        cmd=[sys.executable,'-u','evaluate.py','--source',str(AC_ROOT / 'outputs/table1_ours_v10_compact_fixedtest100'),
             '--v3-root',str(b/'training/critic/stage1'),'--v3-checkpoint',str(baseline),'--v4-output',str(out),
             '--critic-code',str(b),'--easyrec-code',str(AC_ROOT / 'models/easyrec'),'--output',str(r/'evaluation')]
        with (r/'evaluate.log').open('w') as log:subprocess.run(cmd,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state('complete_user_selected',selection=config,evaluation=str(r/'evaluation/summary.json'))
        subprocess.run([sys.executable,'finalize_metadata.py'],cwd=r,check=True)
    except Exception as e:state('failed_user_selected',error=repr(e));raise
if __name__=='__main__':main()
