"""Correct inherited V3 optimization metadata only after the V4 pipeline ends.

The training script preserves V3 architecture args, but also inherited its old
epoch/lr/batch fields. Provenance contains the actual V4 settings. Preserve raw
checkpoints and cryptographically verify every model tensor before updating
artifact hashes. This never changes any score, parameter, epoch or mix choice.
"""
import hashlib,json,os
from pathlib import Path
import torch
from prepare import sha

def weights_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        x=v.detach().cpu().contiguous()
        h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    root=Path(__file__).resolve().parent;out=root.parent/'outputs/bundle_critic_reference_v4'
    if json.loads((root/'pipeline_state.json').read_text())['stage'] not in ['complete','complete_no_accepted_checkpoint','complete_user_selected']:
        raise RuntimeError('Wait for the training and evaluation pipeline to finish')
    report=out/'metadata_finalization.json'
    if report.exists():raise FileExistsError('Metadata already finalized')
    actual=json.loads((out/'provenance.json').read_text())['args'];changes={}
    completed=json.loads((out/'COMPLETE.json').read_text())
    actual['epochs']=completed.get('epochs_completed',actual['epochs'])
    for name in ['best_critic.pt','last_critic.pt']:
        p=out/name
        if not p.exists():continue
        raw=p.with_name(p.stem+'_raw_training.pt')
        if raw.exists():raise FileExistsError(raw)
        ck=torch.load(p,map_location='cpu',weights_only=False);old_sha=sha(p);wh=weights_hash(ck['model_state_dict'])
        old_args=dict(ck['args'])
        ck['args'].update(epochs=actual['epochs'],encoder_lr=actual['encoder_lr'],head_lr=actual['head_lr'],batch=actual['batch_size'],grad_accum=1,seed=42)
        ck['metadata_finalization']={'reason':'Replace inherited V3 optimizer metadata with actual V4 provenance; model parameters unchanged.',
                                     'prior_args':old_args,'raw_checkpoint_sha256':old_sha,'model_tensors_sha256':wh}
        tmp=p.with_suffix('.metadata.tmp');torch.save(ck,tmp)
        check=torch.load(tmp,map_location='cpu',weights_only=False)
        assert weights_hash(check['model_state_dict'])==wh
        p.rename(raw);tmp.rename(p)
        changes[name]={'raw_sha256':old_sha,'final_sha256':sha(p),'model_tensors_sha256':wh,'parameters_unchanged':True}
        del ck,check
    ep=root/'evaluation/summary.json'
    if ep.exists():
        result=json.loads(ep.read_text());c=changes['best_critic.pt']
        assert result['v4_sha256']==c['raw_sha256']
        result['v4_scored_raw_sha256']=result['v4_sha256'];result['v4_sha256']=c['final_sha256']
        result['metadata_finalization']=c
        ep.write_text(json.dumps(result,indent=2)+'\n')
    report.write_text(json.dumps({'changes':changes,'training_selection_and_predictions_unchanged':True},indent=2)+'\n')
    print(report.read_text())
if __name__=='__main__':main()
