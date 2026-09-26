#!/usr/bin/env python3
"""Top-10 tool-bundle beam decoding with a fixed Top-1 CF LLM.

Beam score is negative cumulative action regret: Q(action) minus max Q(state).
This preserves the original greedy path at score zero. Merge permutations by
best path score. No test labels
are passed to encoding, candidate construction, decoding, or ranking.
"""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse
import json
from pathlib import Path
import baseline5_run_infer_rag_gpt as b5

def beam_decode(model,runtime,record,device,b3,width=50,count=10,max_tools=6,min_tools=1):
    import torch
    active=[([],0.0)]; finished={}
    with torch.inference_mode():
        greedy=[]
        for _ in range(max_tools):
            q,cand,cm,sm,lengths=runtime.record_batch([record],[0],[greedy],device)
            values=b3.mask_invalid_actions(model(q,cand,cm,sm),cm,sm,min_bundle_size=min_tools)[0]
            end=len(values)-1
            valid=[j for j in range(lengths[0]) if j not in greedy]
            if len(greedy)>=min_tools: valid.append(end)
            action=valid[int(torch.argmax(values[valid]).item())]
            if action==end: break
            greedy.append(action)
        for step in range(max_tools+1):
            if not active: break
            q,cand,cm,sm,lengths=runtime.record_batch([record],[0]*len(active),[x[0] for x in active],device)
            values=b3.mask_invalid_actions(model(q,cand,cm,sm),cm,sm,min_bundle_size=min_tools)
            new={}
            for bi,(selected,score) in enumerate(active):
                end=values.shape[1]-1
                valid=[j for j in range(lengths[bi]) if j not in selected]
                if len(selected)>=min_tools: valid.append(end)
                if len(selected)>=max_tools: valid=[end]
                scores=values[bi,valid]
                logits=(scores-scores.max()).cpu().tolist()
                for action,logp in zip(valid,logits):
                    seq=selected if action==end else selected+[action]
                    key=tuple(sorted(seq)); total=score+logp
                    pool=finished if action==end else new
                    if key not in pool or total>pool[key][1]: pool[key]=(seq,total)
            active=sorted(new.values(),key=lambda x:(-x[1],tuple(x[0])))[:width]
    ranked=sorted(finished.values(),key=lambda x:(-x[1],tuple(x[0])))
    # Explicitly preserve the old greedy choice, including deterministic ties.
    return [(greedy,0.0)]+[x for x in ranked if set(x[0])!=set(greedy)][:count-1]

def main():
    b5.ensure_env_cuda_library()
    import torch
    import baseline3_train_text2bundle_agent_adapted as b3
    import baseline_top10_common as common
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_dir',type=Path,default=common.ROOT/'outputs/text2bundle_top10_disjoint_train_seed42')
    ap.add_argument('--experiment_dir',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--diagnostic_allow_overlap',action='store_true',
        help='Existing-checkpoint diagnostic only; never a clean test result')
    args=ap.parse_args()
    torch.set_num_threads(4); b3.seed_everything(42)
    manifest,rows=b5.reference_inputs(b5.DEFAULT_SOURCE)
    if args.limit: manifest,rows=manifest[:args.limit],rows[:args.limit]
    config=json.loads((args.train_dir/'config.json').read_text())
    training=b3.load_jsonl(config['data_path'])
    test_qids={str(r['qid']) for r in manifest}; test_queries={r['query'].strip() for r in manifest}
    overlap_rows=sum(str(r.get('qid')) in test_qids or str(r.get('query','')).strip() in test_queries for r in training)
    if overlap_rows and not args.diagnostic_allow_overlap:
        raise ValueError('Training/test query overlap; refusing evaluation')
    train_parsed=b3.parse_rows(training,config['cf_bundle_depth'],config['semantic_tool_depth'],config['cf_llm_depth'])
    train_rows,_,_,_=b3.grouped_split(train_parsed,config['val_ratio'],config['seed'])
    overlap_test_qids=sorted({str(r.qid) for r in train_rows if str(r.qid) in test_qids or r.query.strip() in test_queries})
    # Empty target deliberately prevents test labels entering vocabulary/preparation.
    test_parsed=b3.parse_rows([{'qid':r['dataset_example']['qid'],'query':r['query'],
        'context':r['context']['text'],'target':''} for r in rows],5,25,10)
    tools,_,descriptions,_,_,records,queries=b3.build_vocab_and_records(train_rows,test_parsed,False)
    # Encode only test queries and tools actually exposed to them.
    used=sorted({tid for r in records for tid in r.candidate_tool_ids})
    used_queries=sorted({r.query_index for r in records})
    tm={v:i for i,v in enumerate(used)}; qm={v:i for i,v in enumerate(used_queries)}
    tool_names=[tools[i] for i in used]
    query_texts=[queries[i] for i in used_queries]
    for r in records:
        r.query_index=qm[r.query_index]
        r.candidate_tool_ids=[tm[t] for t in r.candidate_tool_ids]
        r.retrieved_tool_ids=list(r.candidate_tool_ids)
        r.candidate_surface_by_tool_id={tm[t]:v for t,v in r.candidate_surface_by_tool_id.items()}
        if not r.selected_llm or not r.candidate_tool_ids: raise ValueError('Missing test retrieval candidates')
    output=args.experiment_dir; output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda')
    checkpoint=args.train_dir/'best_model.pt'
    run_config={'protocol':'text2bundle_beam_top10_qregret_v2','checkpoint_sha256':b5.file_digest(checkpoint),
        'beam_width':50,'max_tools':6,'min_tools':1,'score':'negative cumulative action Q regret; greedy score zero; END included',
        'llm':'Top1 CF LLM fixed','permutation_dedup':'best path per unordered tool set',
        'test_manifest_sha256':b5.file_digest(b5.DEFAULT_SOURCE/'sample_manifest.jsonl'),
        'training_source_overlap_rows':overlap_rows,'training_overlap_test_qids':overlap_test_qids,
        'diagnostic_only':args.diagnostic_allow_overlap,'n':len(rows)}
    config_hash=b5.digest(run_config)
    b5.write_json(output/'config.json',run_config)
    encoder=b3.EasyRecTextEncoder(config['easyrec_model_dir'],config['easyrec_code_dir'],device,
        config['easyrec_max_length'],config['easyrec_batch_size'],config['fp16_easyrec'])
    qe=encoder.encode(query_texts,'test queries')
    te=encoder.encode([b3.tool_to_text(t,descriptions.get(t,'')) for t in tool_names],'test tools')
    del encoder; torch.cuda.empty_cache()
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=b3.Text2BundlePolicy(**{k:ckpt[k] for k in ['text_dim','model_dim','nhead','state_layers','action_layers','dropout']}).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    runtime=b3.RuntimeData(qe,te,[],records)
    outputs=[]
    for row,record in zip(rows,records):
        bundles=beam_decode(model,runtime,record,device,b3)
        results=[]; seen=set()
        for selected,score in bundles:
            tokens=[record.candidate_surface_by_tool_id[record.candidate_tool_ids[i]] for i in selected]
            key=tuple(sorted(tokens))
            if key in seen: continue
            seen.add(key)
            strict=b3.build_strict_agent_text(record.selected_llm,tokens)
            results.append({'rank':len(results)+1,'llm_token':record.selected_llm,'tool_tokens':tokens,
                            'strict_text':strict,'gen_text':strict,'beam_score':score})
        if not results: raise ValueError('No complete bundles')
        result={'ok':True,'baseline':'Text2Bundle (overlap diagnostic)' if args.diagnostic_allow_overlap else 'Text2Bundle','config_sha256':config_hash,
            'dataset_example':row['dataset_example'],'query':row['query'],'results':results,
            'generation':{'num_return_sequences':len(results),'results':results}}
        b5.write_json(output/'per_sample'/f"{row['dataset_example']['sample_id']}.json",result)
        outputs.append(result)
        print(f'[Text2Bundle] {len(outputs)}/{len(rows)} candidates={len(results)}',flush=True)
    (output/'sample_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    (output/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs))
    common.evaluate(output,outputs,len(rows))

if __name__=='__main__': main()
