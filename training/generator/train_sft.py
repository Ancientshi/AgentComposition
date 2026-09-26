#!/usr/bin/env python3
"""Guarded causal-LM SFT with explicit, prevalidated completion labels."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse, hashlib, json, os, pathlib
from prepare_sft import sha_file, read_rows
from compact_context import VERSION, query_key, components


def validate_data(data_dir,max_seq_len):
    report=json.loads((data_dir/'preparation_report.json').read_text())
    manifest=data_dir/'test_manifest.frozen.jsonl'
    if sha_file(manifest)!=report['test_manifest_sha256']:raise ValueError('Frozen test manifest hash changed')
    if sha_file(pathlib.Path(report['test_manifest']))!=report['test_manifest_sha256']:raise ValueError('Original test manifest changed')
    forbidden={query_key(x['query']) for _,x in read_rows(manifest)};queries={};result={}
    for split in ['train','valid']:
        p=data_dir/f'sft_{split}.jsonl'
        if sha_file(p)!=report['outputs_sha256'][split]:raise ValueError(f'{split} data changed since preparation')
        seen=set();n=0;mi=10**9;ma=0
        for _,row in read_rows(p):
            q=query_key(row['query']);seen.add(q)
            if q in forbidden:raise ValueError('Fixed test query found in training/dev')
            ids,labels=row['input_ids'],row['labels'];n+=1
            if row['completion']!=row['target']:raise ValueError('Stale completion')
            if len(components(row['target'])[1])>report.get('budgets',{}).get('max_tools',6):
                raise ValueError('Target exceeds the configured inference tool limit')
            if not 0<len(ids)<=max_seq_len or len(ids)!=len(labels):raise ValueError('Bad sequence length')
            k=next((i for i,v in enumerate(labels) if v!=-100),None)
            if k is None or k==0 or labels[k:]!=ids[k:]:raise ValueError('Invalid completion supervision')
            if len(row['attention_mask'])!=len(ids) or not all(row['attention_mask']):raise ValueError('Unexpected attention mask')
            mi=min(mi,len(ids)-k);ma=max(ma,len(ids))
        if not n:raise ValueError('Empty split')
        queries[split]=seen;result[split]={'rows':n,'queries':len(seen),'min_answer_tokens':mi,'max_sequence_tokens':ma}
    if queries['train']&queries['valid']:raise ValueError('Train/dev query overlap')
    return report,result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir',type=pathlib.Path,required=True)
    ap.add_argument('--output_dir',type=pathlib.Path,required=True)
    ap.add_argument('--model_name',default=str(AC_BASE_MODEL))
    ap.add_argument('--max_seq_len',type=int,default=8192)
    ap.add_argument('--epochs',type=float,default=1)
    ap.add_argument('--batch_size',type=int,default=1)
    ap.add_argument('--grad_accum',type=int,default=8)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--save_steps',type=int,default=100)
    ap.add_argument('--warmup_steps',type=int,default=100)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--train_embeddings',type=int,choices=[0,1],default=1)
    ap.add_argument('--resume_checkpoint',type=pathlib.Path)
    ap.add_argument('--preflight_only',action='store_true')
    args=ap.parse_args();report,verified=validate_data(args.data_dir,args.max_seq_len)
    if pathlib.Path(args.model_name).resolve()!=pathlib.Path(report['tokenizer']).resolve():
        raise ValueError('Prepared token IDs belong to a different base model/tokenizer')
    if args.preflight_only:
        print(json.dumps({'status':'passed','validated':verified,'test_manifest_sha256':report['test_manifest_sha256']},indent=2));return
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig,Trainer,TrainingArguments,DataCollatorForSeq2Seq,set_seed
    from peft import LoraConfig,get_peft_model
    from accelerate import PartialState
    state=PartialState();out=args.output_dir
    signature={'data_sha256':report['outputs_sha256'],'test_manifest_sha256':report['test_manifest_sha256'],
               'context_version':VERSION,'context_sha256':sha_file(pathlib.Path(__file__).with_name('compact_context.py')),
               'max_prompt_tokens':report['budgets']['prompt'],
               'trainer_sha256':sha_file(__file__),'model_name':args.model_name,
               'training':{k:getattr(args,k) for k in ['max_seq_len','epochs','batch_size','grad_accum','lr','save_steps','warmup_steps','seed','train_embeddings']}}
    if state.is_main_process:
        if args.resume_checkpoint:
            if not args.resume_checkpoint.is_dir() or out.resolve() not in args.resume_checkpoint.resolve().parents:raise ValueError('Resume must be an explicit checkpoint in this output directory')
            if json.loads((out/'training_provenance.json').read_text())!=signature:raise ValueError('Resume data/code/training signature changed')
        else:
            if out.exists() and any(out.iterdir()):raise ValueError('Output directory is not empty. Never automatically resume an old model.')
            out.mkdir(parents=True,exist_ok=True)
            (out/'training_provenance.json').write_text(json.dumps(signature,indent=2))
            (out/'data_validation.json').write_text(json.dumps(verified,indent=2))
    state.wait_for_everyone();set_seed(args.seed)
    config=AutoConfig.from_pretrained(args.model_name,local_files_only=True)
    if args.max_seq_len>config.max_position_embeddings:raise ValueError('Requested sequence exceeds base-model context length')
    tokenizer=AutoTokenizer.from_pretrained(args.model_name,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    ds={}
    for split in ['train','valid']:
        d=load_dataset('json',data_files=str(args.data_dir/f'sft_{split}.jsonl'),split='train')
        ds[split]=d.select_columns(['input_ids','attention_mask','labels'])
    model=AutoModelForCausalLM.from_pretrained(args.model_name,local_files_only=True,torch_dtype=torch.bfloat16)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
                         target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                         modules_to_save=['embed_tokens','lm_head'] if args.train_embeddings else None))
    if state.is_main_process:model.print_trainable_parameters()
    training=TrainingArguments(output_dir=str(out),num_train_epochs=args.epochs,per_device_train_batch_size=args.batch_size,
              per_device_eval_batch_size=1,gradient_accumulation_steps=args.grad_accum,learning_rate=args.lr,
              warmup_steps=args.warmup_steps,logging_steps=10,save_steps=args.save_steps,save_total_limit=2,
              eval_strategy='steps',eval_steps=args.save_steps,bf16=True,gradient_checkpointing=True,
              gradient_checkpointing_kwargs={'use_reentrant':False},ddp_find_unused_parameters=False,
              report_to='none',seed=args.seed,remove_unused_columns=False)
    collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,model=None,padding=True,label_pad_token_id=-100)
    trainer=Trainer(model=model,args=training,train_dataset=ds['train'],eval_dataset=ds['valid'],
                    processing_class=tokenizer,data_collator=collator)
    train=trainer.train(resume_from_checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None)
    trainer.save_model(str(out));trainer.save_state()
    metrics=trainer.evaluate()
    if state.is_main_process:
        tokenizer.save_pretrained(str(out));(out/'metrics.json').write_text(json.dumps({'train':train.metrics,'dev':metrics},indent=2))
        print(f'Saved model to {out}; evaluate recommendation metrics separately on fixed test.')

if __name__=='__main__':main()
