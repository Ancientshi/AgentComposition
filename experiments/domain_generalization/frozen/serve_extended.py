from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import sys,pathlib,hashlib
BASE=pathlib.Path(str(AC_ROOT / 'training/critic/stage1'));sys.path.insert(0,str(BASE))
import serve,compact_input
# Keep the trained serialization byte-for-byte except the cardinality assertion.
source=(BASE/'compact_input.py').read_text();assert source.count('1<=len(tools)<=6')==1
ns={'__name__':'skillbench_compact_input','__file__':str(BASE/'compact_input.py')}
exec(compile(source.replace('1<=len(tools)<=6','1<=len(tools)<=10'),str(BASE/'compact_input.py'),'exec'),ns)
serve.serialize=ns['serialize']
ck=BASE.parent/'outputs/bundle_critic_terra_v3'
s=serve.CompactService(checkpoint_dir=str(ck),easyrec_code_dir=str(AC_ROOT / 'models/easyrec'),device_name='cuda',batch_size=16)
s.metadata['serialization_version']='critic-compact-v3-max10';s.metadata['weights_sha256']=hashlib.sha256((ck/'best_critic.pt').read_bytes()).hexdigest()
s.metadata['adaptation']='original serializer; cardinality assertion6 ->10 only; unchanged512 token budget'
serve.create_app(s).run(host='127.0.0.1',port=8014,threaded=False,use_reloader=False)
