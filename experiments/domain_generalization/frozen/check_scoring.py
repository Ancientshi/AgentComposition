from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json,tempfile,subprocess,sys,math
from pathlib import Path
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);out=p/'outputs/SkillBench_FROZEN_97';w=p/'experiments/domain_generalization/frozen';d=p/'datasets/SkillBench'
 def wr(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
 wr(w/'inputs.json',{'queries':[{'sample_id':'q','query':'synthetic'}],'components':[{'token':'<<a>>','id':'a'},{'token':'<<b>>','id':'b'},{'token':'<<c>>','id':'c'}],'llms':[{'token':'<LLM_x>','id':'x'},{'token':'<LLM_y>','id':'y'}]})
 wr(d/'agents/merge.json',{'gold1':{'M':{'name':'x'},'T':{'tools':['a','b']}},'gold2':{'M':{'name':'y'},'T':{'tools':['c']}}})
 wr(d/'canonical/rankings/merge.json',{'rankings':{'q':['gold1','gold2']}})
 methods=['ours','gpt-5.4-2026-03-05','gpt-5.4-mini-2026-03-17','gpt-5.4-nano-2026-03-17']
 for m in methods:wr(out/m/'per_sample/q.json',{'sample_id':'q','results':[{'llm':'<LLM_x> | description','tools':['<<a>>','<<b>>']},{'llm':'<LLM_y>','tools':['<<c>>']},{'llm':'<LLM_y>','tools':['<<c>>']}],'api':{'returned_model':m,'usage':{}}})
 source=(Path(__file__).parent/'evaluate.py').read_text().replace("ROOT=Path('/root/yunxshi/NIPS2026')",'ROOT=Path('+repr(tmp)+')')
 script=p/'eval.py';script.write_text(source)
 for args,suf,expected in [([], '',0.5),(['--extract-identifiers'],'_identifier_extraction',1.0)]:
  subprocess.run([sys.executable,str(script),*args],check=True,stdout=subprocess.DEVNULL)
  v=json.loads((out/('metrics'+suf+'.json')).read_text())['ours'];met=v['metrics']
  assert met['SetExact-Hit@1']==1 and met['Set-F1@1']==1
  assert met['AgentExact-MRR@10']==expected
  assert v['invalid_component_or_duplicate_slots']==1 and v['missing_slots']==7
  if not suf:
   assert met['AgentExact-Hit@1']==0 and v['invalid_llm_slots_with_valid_sets']==1
   exp=((2/3)/math.log2(2)+1/math.log2(3))/sum(1/math.log2(i+2) for i in range(10))
   assert abs(met['Legacy-RDCR@10']-exp)<1e-12
print('PASS: hand-calculated multi-positive ranks, duplicate positions, invalid-LLM valid sets, missing ranks, discount and literal/extracted identity protocols')
