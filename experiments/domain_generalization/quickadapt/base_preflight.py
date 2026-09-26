from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,hashlib
from pathlib import Path
from transformers import AutoTokenizer
W=Path(__file__).resolve().parent;R=W.parent;B=Path(str(AC_BASE_MODEL))
config=json.loads((R/'checkpoints/generative_v10_compact/adapter_config.json').read_text())
assert config['r']==16 and config['lora_alpha']==32 and config['lora_dropout']==.05
assert set(config['target_modules'])==set(['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
assert set(config['modules_to_save'])=={'embed_tokens','lm_head'}
tok=AutoTokenizer.from_pretrained(B,local_files_only=True)
import sys
sys.path.insert(0,str(R/'training/generator'))
from compact_context import encode_supervision
rows=[];counts={};hashes={};tasks=set()
for f in ['D_target_n3_s42.jsonl','val.jsonl']:
 p=W/'data'/f;current=[json.loads(s) for s in p.read_text().splitlines()];hashes[f]=hashlib.sha256(p.read_bytes()).hexdigest();assert hashes[f]==json.loads((W/'data/audit.json').read_text())['data_hashes'][f]
 for row in current:
  e=encode_supervision(row['prompt'],row['target'],tok,8192)
  assert all(e[k]==row[k] for k in ['input_ids','attention_mask','labels'])
 counts[f]=len(current)
 if f.startswith('D_'):tasks={r['task_id'] for r in current};assert len(tasks)==3 and len(current)==7
result={'status':'PASS','raw_base':str(B),'same_lora_architecture':True,'base_tokenizer_matches_all_train_val_sequences':True,'counts':counts,'train_tasks':sorted(tasks),'hashes':hashes,'no_v10_weight_loading':True}
(W/'BASE_PREFLIGHT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
