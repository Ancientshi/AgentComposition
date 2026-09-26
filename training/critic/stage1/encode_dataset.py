from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import pathlib,json,hashlib,collections,random
from transformers import AutoTokenizer
from compact_input import serialize,VERSION
ROOT=pathlib.Path(__file__).resolve().parent;D=AC_ROOT/'datasets/critic_stage1'
def main():
 tok=AutoTokenizer.from_pretrained(str(AC_EASYREC_MODEL),local_files_only=True);report={'version':VERSION,'splits':{}}
 for split in ['train','valid']:
  output=[];stats=collections.Counter()
  for line in (D/f'cases_{split}.jsonl').read_text().splitlines():
   r=json.loads(line);lab=json.loads((D/'labels'/f"{r['query_hash']}.json").read_text());assert lab['model'].startswith('gpt-5.6-terra')
   cs=[];seen={}
   for c in r['candidates']:
    text,ids,meta=serialize(r['query'],c,r['inventory'],tok)
    sig=(c['llm'],tuple(sorted(c['tools'])));idskey=tuple(ids)
    assert idskey not in seen or seen[idskey]==sig,'Distinct candidates collapsed'
    seen[idskey]=sig
    cs.append({**c,'input_ids':ids,'score':lab['scores'][c['id']]/100,'encoding':meta});stats['candidates']+=1;stats['max_tokens']=max(stats['max_tokens'],len(ids))
   out={'qid':r['qid'],'query_hash':r['query_hash'],'candidates':cs};output.append(out)
  dest=D/f'encoded_{split}.jsonl';dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output))
  report['splits'][split]={'queries':len(output),'stats':stats,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
 (D/'encoding_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
