"""List experiment commands; execute with an explicit --execute flag."""
import argparse, json, os, shlex, subprocess, sys
from string import Template
from .paths import ROOT, BASE_MODEL, EASYREC_MODEL

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment',nargs='?')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args,extra=parser.parse_known_args()
    available=sorted((ROOT/'configs').glob('*.json'))
    if args.list or not args.experiment:
        for path in available:
            config=json.loads(path.read_text());print(f"{path.stem:30} {config['category']}")
        return
    path=ROOT/'configs'/f'{args.experiment}.json'
    if not path.is_file():parser.error('Unknown entry. Use --list.')
    config=json.loads(path.read_text())
    values={**os.environ,'ROOT':str(ROOT),'PYTHON':sys.executable,'BASE_MODEL':str(BASE_MODEL),'EASYREC_MODEL':str(EASYREC_MODEL),'CRITIC_URL':os.environ.get('CRITIC_URL','http://127.0.0.1:8015'),'ADAPT_JOB':os.environ.get('ADAPT_JOB','A_frozen')}
    commands=[[Template(arg).substitute(values) for arg in command] for command in config['commands']]
    if extra and extra[0]=='--':extra=extra[1:]
    if config.get('note'):print(config['note'])
    print('Required inputs:',', '.join(config.get('requires',[])))
    for command in commands:print(shlex.join(command+extra))
    if not args.execute:return
    paths=[ROOT,ROOT/'inference',ROOT/'baselines',ROOT/'models',ROOT/'models/easyrec',ROOT/'training/generator',ROOT/'services/critic']
    environment={**os.environ,'AGENTCOMPOSITION_ROOT':str(ROOT),'BASE_MODEL':str(BASE_MODEL),'EASYREC_MODEL':str(EASYREC_MODEL),'PYTHONPATH':os.pathsep.join(map(str,paths))+os.pathsep+os.environ.get('PYTHONPATH','')}
    for command in commands:subprocess.run(command+extra,cwd=ROOT,env=environment,check=True)
if __name__=='__main__':main()
