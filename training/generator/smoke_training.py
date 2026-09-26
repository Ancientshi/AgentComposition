"""CPU-only compatibility smoke test; never loads or changes the real checkpoint."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json, tempfile, pathlib, math
import torch
from datasets import Dataset
from transformers import AutoTokenizer,LlamaConfig,LlamaForCausalLM,Trainer,TrainingArguments,DataCollatorForSeq2Seq
from peft import LoraConfig,get_peft_model
from prepare_sft import read_rows
root=pathlib.Path(str(AC_ROOT))
tokenizer=AutoTokenizer.from_pretrained(str(AC_BASE_MODEL),local_files_only=True)
# Use a real tokenizer pad id but tiny model-sized synthetic token IDs.
tokenizer.pad_token=tokenizer.convert_ids_to_tokens(0)
records=[]
for _,row in read_rows(root/'datasets/generative_v10_compact/sft_train.jsonl'):
    answer=[t for t in row['labels'] if t!=-100][:24+len(records)*5]
    prompt=row['input_ids'][:12+len(records)*3]
    ids=[3+t%200 for t in prompt+answer]
    records.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prompt)+ids[len(prompt):]})
    if len(records)==2:break
model=LlamaForCausalLM(LlamaConfig(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,pad_token_id=0))
model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],modules_to_save=['embed_tokens','lm_head']))
collator=DataCollatorForSeq2Seq(tokenizer,model=None,padding=True,label_pad_token_id=-100)
batch=collator(records)
assert torch.all((batch['labels']!=-100).sum(1)>0)
assert torch.all(batch['labels'][batch['attention_mask']==0]==-100)
parameter=next(p for n,p in model.named_parameters() if 'lm_head.modules_to_save.default.weight' in n)
before=parameter.detach().clone()
with tempfile.TemporaryDirectory(prefix='generative-v10-cpu-') as out:
    args=TrainingArguments(output_dir=out,max_steps=1,per_device_train_batch_size=2,use_cpu=True,report_to='none',save_strategy='no',logging_steps=1,remove_unused_columns=False)
    trainer=Trainer(model=model,args=args,train_dataset=Dataset.from_list(records),data_collator=collator,processing_class=tokenizer)
    metrics=trainer.train().metrics
assert math.isfinite(metrics['train_loss'])
assert not torch.equal(before,parameter.detach())
print(json.dumps({'status':'passed','real_checkpoint_loaded':False,'device':'cpu','steps':1,'loss':metrics['train_loss'],'weights_updated':True,'answer_tokens_per_row':(batch['labels']!=-100).sum(1).tolist()}))
