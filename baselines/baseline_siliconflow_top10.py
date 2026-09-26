#!/usr/bin/env python3
"""Ranked-list RAG for Llama and the three local-proxy GPT models."""
import argparse
import json
import os
from pathlib import Path
import baseline5_run_infer_rag_gpt as b5

def siliconflow_payload(args, messages):
    return {'model':args.model, 'messages':messages, 'max_tokens':args.max_new_tokens,
            'temperature':0, 'stream':False, 'n':1, 'enable_thinking':False}

b5.build_payload = siliconflow_payload
b5.baseline_name = lambda model: 'RAG-' + model


def main():
    b5.ensure_env_cuda_library()
    import torch
    from transformers import AutoTokenizer
    import baseline_top10_common as common
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['deepseek-ai/DeepSeek-V4-Pro', 'Qwen/Qwen3.8-27B', 'Pro/moonshotai/Kimi-K2.6'])
    ap.add_argument('--source_experiment', type=Path, default=b5.DEFAULT_SOURCE)
    ap.add_argument('--experiment_dir', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--api_base_url', default='http://127.0.0.1:18082/v1')
    ap.add_argument('--api_mode', default='chat_completions', choices=['chat_completions'])
    ap.add_argument('--http_proxy', default='')
    ap.add_argument('--api_timeout', type=float, default=300)
    ap.add_argument('--api_retries', type=int, default=2)
    ap.add_argument('--max_new_tokens', type=int, default=2048)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    manifest, rows = b5.reference_inputs(args.source_experiment)
    if args.limit: rows, manifest = rows[:args.limit], manifest[:args.limit]
    output = args.experiment_dir
    output.mkdir(parents=True, exist_ok=True)
    b4, sota = common.modules()
    model_dir = rows[0]['config']['model_dir']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prepared = [common.prompt(r,b4,sota,tokenizer) for r in rows]
    config = {'provider': 'siliconflow', 'api_mode':args.api_mode, 'api_base_url':args.api_base_url, 'enable_thinking':False, 'temperature':0, 'protocol': 'ranked_list_top10_v1', 'model':args.model, 'seed':42,
              'input_budget':6000, 'max_new_tokens':args.max_new_tokens, 'max_repairs':2,
              'source_sha256': b5.file_digest(args.source_experiment/'results.jsonl'),
              'messages_sha256':[p[3]['messages_sha256'] for p in prepared],
              'source_code_sha256': {p.name:b5.file_digest(p) for p in [Path(__file__),Path(common.__file__)]}}
    config_hash = b5.digest(config)
    cp = output/'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config: raise ValueError('Resume config changed')
    b5.write_json(cp,config)
    (output/'sample_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    if args.dry_run:
        print(json.dumps({'prepared':len(rows),'token_max':max(x[3]['input_token_count'] for x in prepared),
                          'truncated':sum(x[3]['evidence_truncated'] for x in prepared)}),flush=True)
        return
    model = None
    if args.model == 'llama':
        tokenizer,model,_ = b4._load_base_model(model_dir,torch_dtype='bfloat16',device_map='auto')
    records=[]
    for row,(messages,llms,tools,audit) in zip(rows,prepared):
        sid=row['dataset_example']['sample_id']
        path=output/'per_sample'/f'{sid}.json'
        if path.exists():
            record=json.loads(path.read_text())
            if record['config_sha256'] != config_hash: raise ValueError('Cached config mismatch')
            records.append(record)
            continue
        results=[]; attempts=[]
        for attempt in range(3):
            sent=[dict(m) for m in messages]
            if attempt:
                excluded='\n'.join(r['strict_text'] for r in results)
                sent[-1]['content'] += (f'\nADDITIONAL CANDIDATES REQUIRED\nReturn exactly {10-len(results)} further distinct configurations, in descending suitability. '
                    'Do not repeat any configuration below. Tool reordering is a duplicate.\n'+excluded)
            if model is None:
                text,api=b5.call_api(args,sent)
                if api.get('refusals'): raise RuntimeError('Model refused the recommendation task')
            else:
                encoded=tokenizer.apply_chat_template(sent,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
                if encoded['input_ids'].shape[1]+args.max_new_tokens > model.config.max_position_embeddings:
                    # Repair lists can be longer; keep all input and use remaining output capacity.
                    max_tokens=model.config.max_position_embeddings-encoded['input_ids'].shape[1]
                else: max_tokens=args.max_new_tokens
                if max_tokens < 256: raise ValueError('Repair input exhausts context window')
                encoded={k:v.to(model.device) for k,v in encoded.items()}
                with torch.inference_mode():
                    generated=model.generate(**encoded,max_new_tokens=max_tokens,do_sample=False,num_beams=1,
                        num_return_sequences=1,pad_token_id=tokenizer.pad_token_id)
                text=tokenizer.decode(generated[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)
                api={'requested_model':model_dir,'input_tokens':encoded['input_ids'].shape[1],
                     'output_tokens':generated.shape[1]-encoded['input_ids'].shape[1]}
            results,rejected=common.parse_candidates(text,llms,tools,b4,sota,results)
            attempts.append({'raw_output':text,'api':api,'rejected':rejected,'messages_sha256':b5.digest(sent)})
            if len(results)==10: break
        if not results:
            b5.write_json(output/'failures'/f'{sid}.json',{'sample_id':sid,'attempts':attempts,'audit':audit})
            raise ValueError(f'No parseable candidates for {sid}; raw attempts saved')
        record={'ok':True,'baseline':'RAG-Llama' if args.model=='llama' else b5.baseline_name(args.model),
                'config_sha256':config_hash,'dataset_example':row['dataset_example'], 'query':row['query'],
                'prompt':{'messages':messages,**audit},'attempts':attempts,'results':results,
                'generation':{'results':results,'num_return_sequences':len(results)},
                'ranking':'model list order, followed by ordered additional candidates if needed'}
        b5.write_json(path,record)
        records.append(record)
        print(f'[{args.model}] {len(records)}/{len(rows)} {sid} candidates={len(results)} calls={len(attempts)}',flush=True)
    (output/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    common.evaluate(output,records,len(rows))

if __name__=='__main__': main()
