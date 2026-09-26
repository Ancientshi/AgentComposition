#!/usr/bin/env python3
"""Rebuild traceable PartIII-LLM + PartII-tool supervision; freeze Table1 test."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import argparse, collections, hashlib, json, pathlib, re, shutil, statistics, time
from compact_context import build_prompt, components, encode_supervision, query_key, query_hash, VERSION


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read_rows(path):
    with open(path) as f:
        for i,line in enumerate(f,1):
            if line.strip(): yield i,json.loads(line)


def tool_key(t):
    t=t.strip().strip('<>')
    return t[5:] if t.startswith('TOOL_') else t


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path(str(AC_ROOT)))
    ap.add_argument('--output',type=pathlib.Path)
    ap.add_argument('--tokenizer',default=str(AC_BASE_MODEL))
    ap.add_argument('--max_prompt_tokens',type=int,default=6144)
    ap.add_argument('--max_seq_len',type=int,default=8192)
    ap.add_argument('--max_tools',type=int,default=6)
    ap.add_argument('--dev_fraction',type=float,default=.1)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args();r=args.root;out=args.output or r/'datasets/generative_v10_compact'
    if out.exists() and any(out.iterdir()):raise SystemExit('Output is not empty; choose a fresh directory.')
    if not 0<args.dev_fraction<1:raise ValueError('dev_fraction must be between zero and one')
    out.mkdir(parents=True,exist_ok=True)
    manifest=r/'outputs/table1_ours_seed42_n100_beam1-2_critic05_finalonly/sample_manifest.jsonl'
    frozen=[x for _,x in read_rows(manifest)]
    if len(frozen)!=100 or len({x['sample_id'] for x in frozen})!=100:raise ValueError('Expected fixed 100 test samples')
    test_hash=sha_file(manifest);shutil.copyfile(manifest,out/'test_manifest.frozen.jsonl')
    test_keys={query_key(x['query']) for x in frozen};test_qids={x['qid'] for x in frozen}
    sources=[r/'datasets/generative_v9_sft'/name for name in ['sft_train_PartII.jsonl','sft_valid_PartII.jsonl']]
    p3a=r/'datasets/PartIII/agents/merge.json';p3q=r/'datasets/PartIII/questions/merge.json';p3r=r/'datasets/PartIII/rankings/merge.json'
    agents=json.load(p3a.open());questions=json.load(p3q.open());rankings=json.load(p3r.open())['rankings']
    p2a=r/'datasets/PartII/agents/merge.json';p2q=r/'datasets/PartII/questions/merge.json';p2r=r/'datasets/PartII/rankings/merge.json'
    a2=json.load(p2a.open());q2=json.load(p2q.open());r2=json.load(p2r.open())['rankings'];byquery=collections.defaultdict(list)
    for qid,ids in r2.items():
        if ids:
            aid=ids[0];raw=a2[aid].get('T',{}).get('tools',[])
            byquery[query_key(q2[qid]['input'])].append((qid,aid,frozenset(tool_key(t) for t in raw)))
    # Exclude all variants sharing a test qid, then use query groups for dev split.
    forbidden=set(test_keys)
    for path in sources:
        for _,x in read_rows(path):
            if x.get('qid') in test_qids:forbidden.add(query_key(x['query']))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    counts=collections.Counter();seen=set();groups={'train':set(),'valid':set()};sizes=collections.defaultdict(list)
    output_files={s:(out/f'sft_{s}.jsonl').open('w') for s in groups};reject=(out/'excluded.jsonl').open('w')
    before_sample=[];after_sample=[];examples=[];start=time.time()
    def skip(reason,x,path,line):
        counts['excluded_'+reason]+=1
        reject.write(json.dumps({'reason':reason,'source':str(path),'line':line,'qid':x.get('qid'),'query_hash':query_hash(x['query'])})+'\n')
    try:
        for path in sources:
            for line,x in read_rows(path):
                counts['input_rows']+=1
                if args.limit and counts['input_rows']>args.limit:break
                q=x['query'];qk=query_key(q);qid=x.get('qid');aid=x.get('agent_id');target=x['target'].strip()
                if qk in forbidden:skip('fixed_test_query_or_qid',x,path,line);continue
                lm,ts=components(target)
                if len(lm)!=1 or not ts or not target.endswith('<SPECIAL_END>'):skip('malformed_target',x,path,line);continue
                if lm[0].lower() in {'<llm_default>','<llm_default_llm>','<llm_unk>','<llm_unknown>','<llm_human>','<llm_backbone_llm>','<llm_backbone>'}:skip('placeholder_llm',x,path,line);continue
                m=agents.get(aid,{}).get('M',{});model=m.get('name') if isinstance(m,dict) else m
                # Match v8 normalize_raw_name: each whitespace character becomes '_'.
                normalized_model=''.join('_' if ch.isspace() else ch for ch in str(model).strip())
                expected=f'<LLM_{normalized_model}>'
                if not model or expected!=lm[0] or qk!=query_key(questions.get(qid,{}).get('input','')) or aid not in rankings.get(qid,[]):
                    skip('unverified_partiii_llm',x,path,line);continue
                goldtools=frozenset(tool_key(t) for t in ts);matches=byquery.get(qk,[])
                if not matches or len({v[2] for v in matches})!=1 or matches[0][2]!=goldtools:
                    skip('unverified_or_ambiguous_partii_tools',x,path,line);continue
                # Completion is reconstructed, never copied from stale legacy fields.
                target=' '.join([lm[0],'<TOOL_SEP>']+list(dict.fromkeys(ts))+['<SPECIAL_END>'])
                dedup=(qk,target)
                if dedup in seen:skip('duplicate_query_target',x,path,line);continue
                context=x['context'];available=set(sum(components(context),[]))
                missing=[t for t in lm+ts if t not in available]
                if missing:skip('target_not_in_cached_candidates',x,path,line);continue
                try:
                    prompt,stats=build_prompt(context,q,tok,args.max_prompt_tokens)
                    encoded=encode_supervision(prompt,target,tok,args.max_seq_len)
                except ValueError as e:skip('budget_or_encoding',x,path,line);continue
                if len(set(ts))>args.max_tools:skip('tool_count_limit',x,path,line);continue
                seen.add(dedup)
                # query hash, not row shuffling: all positive agents for one query stay together.
                bucket=int(hashlib.sha256((str(args.seed)+'\0'+qk).encode()).hexdigest()[:16],16)/2**64
                split='valid' if bucket<args.dev_fraction else 'train'
                row={'qid':qid,'agent_id':aid,'query':q,'query_hash':query_hash(q),'target':target,'prompt':prompt,'completion':target,
                     'split':split,'provenance':{'llm_source':str(p3a),'llm_agent_id':aid,'llm_rank':rankings[qid].index(aid)+1,
                     'tool_source':str(p2a),'partii_question_ids':[v[0] for v in matches],'partii_agent_ids':sorted({v[1] for v in matches}),
                     'cached_context_source':str(path),'source_line':line,'historical_context_may_include_gold_injection':True},
                     'compression':stats,**encoded}
                output_files[split].write(json.dumps(row,ensure_ascii=False)+'\n');groups[split].add(qk);counts[split+'_rows']+=1
                sizes[split+'_prompt'].append(stats['prompt_tokens']);sizes[split+'_total'].append(len(encoded['input_ids']))
                sizes[split+'_supervised'].append(sum(t!=-100 for t in encoded['labels']))
                counts['stale_completion_rebuilt']+=not x['completion'].startswith(x['target'])
                if len(before_sample)<256:
                    before_sample.append(len(tok.encode(x['prompt'],add_special_tokens=True)));after_sample.append(stats['prompt_tokens'])
                if len(examples)<3:examples.append({k:row[k] for k in ['qid','target','provenance','compression']})
                if (counts['train_rows']+counts['valid_rows'])%1000==0:print(json.dumps({'progress':dict(counts),'seconds':round(time.time()-start)}),flush=True)
            if args.limit and counts['input_rows']>args.limit:break
    finally:
        for f in output_files.values():f.close()
        reject.close()
    assert not groups['train']&groups['valid']
    assert not (groups['train']|groups['valid'])&forbidden
    assert sha_file(manifest)==test_hash==sha_file(out/'test_manifest.frozen.jsonl')
    def stats(v):return {'min':min(v),'median':statistics.median(v),'max':max(v),'mean':statistics.mean(v)} if v else {}
    summary={'version':VERSION,'counts':dict(counts),'unique_queries':{k:len(v) for k,v in groups.items()},
             'test_manifest':str(manifest),'test_manifest_sha256':test_hash,'test_samples':100,
             'test_overlap_train':0,'test_overlap_valid':0,'train_valid_query_overlap':0,
             'budgets':{'prompt':args.max_prompt_tokens,'sequence':args.max_seq_len,'tool_description':32,'llm_description':48,'max_tools':args.max_tools},
             'token_stats':{k:stats(v) for k,v in sizes.items()},'first256_token_comparison':{'before':stats(before_sample),'after':stats(after_sample)},
             'source_sha256':{str(p):sha_file(p) for p in sources+[p3a,p3q,p3r,p2a,p2q,p2r]},
             'outputs_sha256':{s:sha_file(out/f'sft_{s}.jsonl') for s in groups},'tokenizer':args.tokenizer,
             'notes':['LLM labels are inherited PartIII labels, not newly measured backbone performance.',
                      'Historical cached contexts may contain earlier gold injection; no new gold injection is performed.',
                      'Unreachable targets and placeholder LLMs are excluded; see excluded.jsonl.'],
             'examples':examples,'seconds':round(time.time()-start,2)}
    (out/'preparation_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if not counts['train_rows'] or not counts['valid_rows']:raise SystemExit('Empty training or validation split')

if __name__=='__main__':main()
