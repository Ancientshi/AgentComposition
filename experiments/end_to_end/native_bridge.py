"""Local, allowlisted public-API counterparts with immutable response caching."""
import difflib
import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'native_cache';CACHE.mkdir(exist_ok=True)
LOCK=threading.Lock()
KEY_LOCKS={}


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'AgentRecommendationResearch/1.0'})
    with urllib.request.urlopen(request,timeout=20) as f:
        return json.load(f)


def native(name,a):
    urls=[]
    def get(base,params=None):
        url=base+('?' + urllib.parse.urlencode(params) if params else '')
        urls.append(url)
        return fetch(url)
    if name.startswith('Currency'):
        source=a.get('from','USD').upper();target=a.get('to','EUR').upper()
        if 'Free Exchange Rates' in name:
            return get('https://api.frankfurter.dev/v1/latest',{'base':'USD'}),urls
        x=get('https://api.frankfurter.dev/v1/latest',{'base':source,'symbols':target})
        amount=float(a.get('amount',a.get('q',1)))
        rate=x['rates'][target]
        return {'base':source,'target':target,'amount':amount,'rate':rate,'converted_amount':amount*rate,'date':x['date']},urls
    if name=='Open Library&&Search Title':
        return get('https://openlibrary.org/search.json',{'title':a['title'],'limit':10,
                    'fields':'key,title,author_name,first_publish_year,edition_count'}),urls
    if name=='Cat Facts&&Facts':return get('https://catfact.ninja/facts',{'limit':5}),urls
    if name=='Joke Test&&/random_joke':return get('https://official-joke-api.appspot.com/random_joke'),urls
    if name=='Deezer&&Playlist':return get('https://api.deezer.com/playlist/'+urllib.parse.quote(str(a['id']),safe='')),urls
    if name=='Steam&&Search':return get('https://store.steampowered.com/api/storesearch/',{'term':a['term'],'l':'english','cc':'us'}),urls
    if name=='Whois Lookup_v3&&Check Similarity':
        return {'domain1':a['domain1'],'domain2':a['domain2'],
                'normalized_levenshtein_alternative_sequence_match_ratio':difflib.SequenceMatcher(None,a['domain1'].lower(),a['domain2'].lower()).ratio()},urls
    if name=='Whois Lookup_v3&&DNS Lookup':return get('https://dns.google/resolve',{'name':a['domain'],'type':a.get('rtype','A')}),urls
    if name=='Whois Lookup_v3&&NS Lookup':return get('https://dns.google/resolve',{'name':a['search'],'type':'A'}),urls
    return None,[]


def call(name,args):
    key=hashlib.sha256(json.dumps([name,args],sort_keys=True).encode()).hexdigest()
    p=CACHE/(key+'.json')
    with LOCK:
        key_lock=KEY_LOCKS.setdefault(key,threading.Lock())
    with key_lock:
        if p.exists():return json.loads(p.read_text())
        try:
            value,urls=native(name,args)
            if value is None:return {'status':'unavailable','reason':'No credential-free deployed counterpart'}
            result={'status':'ok','mode':'native_counterpart','data':value,'sources':urls,'retrieved_at_unix':time.time()}
        except Exception as e:
            result={'status':'unavailable','reason':type(e).__name__+': '+str(e)[:200]}
        p.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        return result


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            assert self.path=='/call'
            n=int(self.headers['Content-Length']);assert 0<n<100000
            body=json.loads(self.rfile.read(n));x=call(body['name'],body['arguments'])
            b=json.dumps(x,ensure_ascii=False).encode();self.send_response(200)
        except Exception as e:
            b=json.dumps({'error':type(e).__name__}).encode();self.send_response(400)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def log_message(self,*args):pass


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',18186),Handler).serve_forever()
