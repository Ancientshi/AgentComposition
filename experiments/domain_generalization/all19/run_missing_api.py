from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import concurrent.futures,json,os,re,time,urllib.request,urllib.error
from pathlib import Path
R=Path(str(AC_ROOT));O=R/'outputs/SkillsBench_TEST19_20260923'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def run(task):
 m=task['model'];sid=task['sample_id'];p=O/m/'per_sample'/f'{sid}.json'
 if p.exists():return 'cached'
 gpt=m.startswith('gpt-');prefix='GPT' if gpt else 'SILICONFLOW'
 url=os.environ.get(prefix+'_BASE_URL',f'http://127.0.0.1:{18080 if gpt else 18082}/v1').rstrip('/')+'/chat/completions'
 payload={'model':m,'messages':task['messages'],'temperature':0,'stream':False,'n':1}
 payload.update({'max_completion_tokens':5000,'reasoning_effort':'none','store':False} if gpt else {'max_tokens':5000,'enable_thinking':False})
 headers={'Content-Type':'application/json'};key=os.environ.get(prefix+'_API_KEY','')
 if key:headers['Authorization']='Bearer '+key
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 start=time.time()
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers)
   with opener.open(req,timeout=300) as response:body=json.load(response)
   break
  except urllib.error.HTTPError as e:
   if e.code not in [408,429,500,502,503,504] or attempt==2:raise RuntimeError('HTTP '+str(e.code)) from None
  except (urllib.error.URLError,TimeoutError,ConnectionError):
   if attempt==2:raise RuntimeError('API connection failed after3 transport attempts') from None
  time.sleep(2**attempt)
 choices=body.get('choices');assert choices and len(choices)==1,'Missing choices'
 text=choices[0]['message'].get('content') or ''
 row={'sample_id':sid,'raw_output':text,'messages':task['messages'],'results':[],'elapsed_sec':time.time()-start,'api':{'requested_model':m,'returned_model':body.get('model'),'usage':body.get('usage'),'finish_reason':choices[0].get('finish_reason'),'response_id':body.get('id')}}
 write(p,row);assert body.get('model')==m,'Returned model mismatch'
 try:
  ranking=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()))['ranking'];assert isinstance(ranking,list);row['results']=ranking[:10]
 except Exception as e:row['format_error']=str(e)
 write(p,row);return 'complete'
tasks=read(O/'missing_api_requests.json');errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('EVAL_API_WORKERS','12'))) as pool:
 fs={pool.submit(run,t):t for t in tasks}
 for i,f in enumerate(concurrent.futures.as_completed(fs),1):
  t=fs[f]
  try:status=f.result()
  except Exception as e:errors.append({'model':t['model'],'sample_id':t['sample_id'],'error':str(e)});status='blocked'
  print(i,len(tasks),t['model'],t['sample_id'],status,flush=True)
write(O/'new_api_status.json',{'expected':len(tasks),'errors':errors,'failed_transport_policy':'Do not manufacture model failures for unavailable authentication or endpoints; report incomplete until requests can execute.'})
