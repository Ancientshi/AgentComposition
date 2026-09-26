"""Shared candidate-preserving 512-token representation; no target/teacher fields."""
import re
VERSION='critic-compact-v3'
TOK=re.compile(r'<<[^<>\n]+>>|<LLM_[^<>\n]+>|<TOOL_[^<>\n]+>')
SPECIAL={'<TOOL_SEP>','<TOOL_EMPTY>','<SPECIAL_END>'}
def tokens(s):return list(dict.fromkeys(t for t in TOK.findall(s) if t not in SPECIAL))
def clip(s,tok,n,tail=False):
 ids=tok.encode(str(s),add_special_tokens=False)
 if len(ids)<=n:return str(s)
 if n<=0:return ''
 if tail and n>=16:
  a=n*3//4;return tok.decode(ids[:a],skip_special_tokens=True)+' ... '+tok.decode(ids[-(n-a):],skip_special_tokens=True)
 return tok.decode(ids[:n],skip_special_tokens=True)
def clean(s):
 s=re.sub(r'https?://\S+','',str(s));s=re.sub(r'\s+',' ',s).strip()
 if 'gold' in s.lower() and any(x in s.lower() for x in ['inject','target','supervis']):return ''
 if 'Api Description:' in s:s=s.rsplit('Api Description:',1)[-1]
 return s

def compact_inventory(text):
 section='';llms=[];tools=[];desc={};bundles=[]
 for line in text.splitlines():
  if line.startswith('LLM candidates'):section='llm';continue
  if line.startswith('Retrieved tool bundles'):section='bundle';continue
  if line.startswith('Tool candidates'):section='tool';continue
  if line.startswith('###'):section='';continue
  ts=tokens(line)
  if section=='llm':llms.extend(x for x in ts if x.startswith('<LLM_'))
  if section=='tool':tools.extend(x for x in ts if not x.startswith('<LLM_'))
  if section=='bundle' and ts:bundles.append([x for x in ts if not x.startswith('<LLM_')])
  if section in ['llm','tool'] and ts and ' | ' in line:desc[ts[0]]=clean(line.split(' | ',1)[1])
 return {'llms':list(dict.fromkeys(llms)),'tools':list(dict.fromkeys(tools)),'descriptions':desc,'bundles':bundles}

def serialize(query,candidate,inventory,tok,max_length=512):
 llm=candidate['llm'];tools=sorted(set(candidate['tools']));assert llm.startswith('<LLM_') and 1<=len(tools)<=6
 # Exact IDs are never normalized or truncated; tool order is not a feature.
 candidate_text='[CANDIDATE]\nLLM: '+llm+'\nTOOLS:\n'+'\n'.join(tools)
 descriptions=inventory['descriptions'];info=[clean(descriptions.get(x,'')) for x in [llm]+tools]
 for qn in [128,112,96,80,64]:
  q=clip(query,tok,qn,tail=True)
  for en in [40,32,24,16,8,0]:
   ev='\n[EVIDENCE]\n'+'\n'.join(('LLM' if i==0 else 'T'+str(i))+': '+(clip(s,tok,en) if s and en else 'unavailable') for i,s in enumerate(info))
   text=candidate_text+'\n[QUERY]\n'+q+ev
   ids=tok.encode(text,add_special_tokens=True)
   if len(ids)<=max_length:
    assert all(x in text for x in [llm]+tools)
    return text,ids,{'query_budget':qn,'evidence_budget_each':en,'length':len(ids),'version':VERSION}
 raise ValueError('Candidate identifiers and minimum query exceed input budget')
