"""Build V4 from clean V3 splits and source references; zero new LLM calls."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
from core import VERSION, check_splits, identities, make_case, norm, parse_target, signature

def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v3-root',type=Path,required=True)
    ap.add_argument('--source-data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite V4 prepared data')
    cases={s:list(jsonl(AC_ROOT/'datasets/critic_stage1'/f'cases_{s}.jsonl')) for s in ['train','valid']}
    manifest=args.source_data/'test_manifest.frozen.jsonl'
    test=list(jsonl(manifest));check_splits(cases['train'],cases['valid'],test)
    wanted={(r['qid'],norm(r['query'])) for rs in cases.values() for r in rs}
    refs=defaultdict(dict);sources={str(manifest):sha(manifest)}
    for split in ['train','valid']:
        p=args.source_data/f'sft_{split}.jsonl';sources[str(p)]=sha(p)
        for r in jsonl(p):
            key=(r['qid'],norm(r['query']))
            if key in wanted:
                gold=parse_target(r['target']);refs[key][signature(gold)]=gold
    missing=wanted-set(refs)
    if missing:raise ValueError(f'Missing source references for {len(missing)} queries')
    args.output.mkdir(parents=True)
    report={'version':VERSION,'new_llm_calls':0,'reference_policy':'Single complete reference at a time, never union; additions are reference-extraneous, not proven semantically useless.',
            'test_policy':'Only test qid/query exclusion; test targets never used for constructing training or validation cases.',
            'source_sha256':sources,'splits':{}}
    for split,rs in cases.items():
        totals=Counter();out=args.output/f'cases_{split}.jsonl'
        with out.open('w') as f:
            for r in rs:
                labelpath=args.v3_root/'data/labels'/f"{r['query_hash']}.json"
                label=json.loads(labelpath.read_text())
                if not label['model'].startswith('gpt-5.6-terra'):raise ValueError('Unexpected V3 teacher')
                if set(label['scores']) != {c['id'] for c in r['candidates']}:raise ValueError('Teacher cache mismatch')
                cs=make_case(r,list(refs[(r['qid'],norm(r['query']))].values()),label)
                if not cs['pairs']:raise ValueError('Query has no training pairs')
                totals.update(cs['selected_pair_counts']);totals['queries']+=1;totals['candidates']+=len(cs['candidates'])
                f.write(json.dumps(cs,ensure_ascii=False)+'\n')
        report['splits'][split]={'counts':dict(totals),'sha256':sha(out)}
        report['source_sha256'][str(AC_ROOT/'datasets/critic_stage1'/f'cases_{split}.jsonl')]=sha(AC_ROOT/'datasets/critic_stage1'/f'cases_{split}.jsonl')
    (args.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
