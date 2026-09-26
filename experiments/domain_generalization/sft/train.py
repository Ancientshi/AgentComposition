from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import os,json,hashlib,time
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,DataCollatorForSeq2Seq,set_seed
from datasets import load_dataset
from peft import PeftModel
from accelerate import PartialState
R=Path(str(AC_ROOT));D=R/'datasets/SkillBench_SFT_v1';O=R/'checkpoints/skillbench_v1_from_v10';INIT=R/'checkpoints/generative_v10_compact';BASE=str(AC_BASE_MODEL)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=PartialState();set_seed(42);report=json.loads((D/'preparation_report.json').read_text())
assert sha(D/'train.jsonl')==report['train_sha256'] and sha(D/'val.jsonl')==report['val_sha256']
manifest=json.loads((D/'split_manifest.json').read_text());forbidden=set(manifest['query_ids']['test']);seen={}
for split in ['train','val']:
 rows=[json.loads(s) for s in (D/f'{split}.jsonl').read_text().splitlines()];seen[split]={r['sample_id'] for r in rows};assert not seen[split]&forbidden
 for r in rows:
  assert len(r['input_ids'])==len(r['labels'])<=8192
  k=next(i for i,v in enumerate(r['labels']) if v!=-100);assert k>0 and r['labels'][k:]==r['input_ids'][k:]
assert not seen['train']&seen['val']
if state.is_main_process:
 assert not O.exists() or not any(O.iterdir()),'Do not overwrite or silently resume a trained checkpoint'
 O.mkdir(parents=True,exist_ok=True)
 (O/'protocol.json').write_text(json.dumps({'initial_checkpoint':str(INIT),'base_model':BASE,'training':'continue existing LoRA weights; freeze all non-LoRA parameters including existing trained embeddings/lm_head','epochs':3,'learning_rate':5e-5,'effective_batch_size':8,'seed':42,'selection':'minimum validation completion loss at epoch boundary','critic':False,'data':report,'train_code_sha256':sha(Path(__file__))},indent=2))
state.wait_for_everyone();tokenizer=AutoTokenizer.from_pretrained(INIT,local_files_only=True);tokenizer.pad_token=tokenizer.eos_token
base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
model=PeftModel.from_pretrained(base,INIT,is_trainable=True)
for name,param in model.named_parameters():param.requires_grad_('lora_' in name)
model.config.use_cache=False
if state.is_main_process:model.print_trainable_parameters()
ds={s:load_dataset('json',data_files=str(D/f'{s}.jsonl'),split='train').select_columns(['input_ids','attention_mask','labels']) for s in ['train','val']}
args=TrainingArguments(output_dir=str(O),num_train_epochs=3,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=2,learning_rate=5e-5,warmup_ratio=.1,lr_scheduler_type='cosine',logging_steps=1,save_strategy='epoch',eval_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},ddp_find_unused_parameters=False,report_to='none',seed=42,remove_unused_columns=False,dataloader_num_workers=0)
trainer=Trainer(model=model,args=args,train_dataset=ds['train'],eval_dataset=ds['val'],processing_class=tokenizer,data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=None,padding=True,label_pad_token_id=-100))
start=time.time();result=trainer.train();trainer.save_model(str(O/'best'));tokenizer.save_pretrained(O/'best') if state.is_main_process else None
if state.is_main_process:
 trainer.save_state();(O/'COMPLETE.json').write_text(json.dumps({'status':'complete','best_checkpoint':trainer.state.best_model_checkpoint,'best_eval_loss':trainer.state.best_metric,'training_metrics':result.metrics,'elapsed_sec':time.time()-start,'world_size':state.num_processes},indent=2));print('TRAINING COMPLETE',flush=True)
