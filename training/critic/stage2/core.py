"""Reference-aligned V4 augmentation and metrics; no model/API dependencies."""
import hashlib
import itertools
import math
import random
import re
from collections import Counter

VERSION = 'critic-v4-reference-redundancy-v1'
TOKEN = re.compile(r'<<[^<>\n]+>>|<LLM_[^<>\n]+>|<TOOL_[^<>\n]+>')
SPECIAL = {'<TOOL_SEP>', '<TOOL_EMPTY>'}
PAIR_LIMITS = {'redundancy': 32, 'missing_required': 32, 'replacement': 16, 'teacher_llm': 32}
MIX_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

def blend_scores(baseline, trained, alpha):
    """Candidate-pool z normalization; alpha is selected on validation only."""
    if len(baseline)!=len(trained) or not 0<=alpha<=1:
        raise ValueError('Invalid blend inputs')
    def z(xs):
        if not xs or any(not math.isfinite(x) for x in xs):raise ValueError('Invalid scores')
        m=sum(xs)/len(xs);sd=math.sqrt(sum((x-m)**2 for x in xs)/len(xs))
        return [(x-m)/sd for x in xs] if sd>1e-12 else [0.]*len(xs)
    a,b=z(baseline),z(trained)
    return [(1-alpha)*x+alpha*y for x,y in zip(a,b)]

def norm(text):
    return ' '.join(str(text).split())

def signature(candidate):
    return candidate['llm'], tuple(sorted(set(candidate['tools'])))

def parse_target(text):
    ts = list(dict.fromkeys(TOKEN.findall(text)))
    llms = [t for t in ts if t.startswith('<LLM_')]
    if len(llms) != 1:
        raise ValueError('Expected exactly one reference LLM')
    return {'llm': llms[0], 'tools': sorted(t for t in ts if t not in SPECIAL and not t.startswith('<LLM_'))}

def metrics(candidate, references):
    p = set(candidate['tools'])
    resolved = [g for g in references if g['llm']]
    if not resolved:
        raise ValueError('No resolved reference')
    overlap = [(len(p & set(g['tools'])), int(candidate['llm'] == g['llm']), len(set(g['tools']))) for g in resolved]
    return {'cp': max((h+m)/(1+len(p)) for h,m,g in overlap),
            'cr': max((h+m)/(1+g) for h,m,g in overlap),
            'tp': max(h/len(p) if p else float(g==0) for h,m,g in overlap),
            'tr': max(h/g if g else 1. for h,m,g in overlap),
            'cf1': max(2*(h+m)/(2+len(p)+g) for h,m,g in overlap),
            'tool_hit': int(any(h == g for h,m,g in overlap)),
            'agent_hit': int(any(h == g and m for h,m,g in overlap))}

def ranked_metrics(candidates, scores, references):
    if len(candidates) != len(scores) or any(not math.isfinite(s) for s in scores):
        raise ValueError('Missing/nonfinite scores')
    if len({signature(c) for c in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate configuration')
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], signature(candidates[i])))[:10]
    ms = [metrics(candidates[i], references) for i in order]
    weights = [1/math.log2(i+2) for i in range(10)]
    return {'RDCP@10': sum(w*m['cp'] for w,m in zip(weights,ms))/sum(weights),
            'RDCR@10': sum(w*m['cr'] for w,m in zip(weights,ms))/sum(weights),
            'Tool-Hit@10': max((m['tool_hit'] for m in ms), default=0),
            'CR-Hit@10': max((m['agent_hit'] for m in ms), default=0),
            'CR-MRR@10': next((1/(i+1) for i,m in enumerate(ms) if m['agent_hit']),0.),
            'RDCF1@10': sum(w*m['cf1'] for w,m in zip(weights,ms))/sum(weights),
            'CompR@1': ms[0]['cr'] if ms else 0.,
            'CompP@1': ms[0]['cp'] if ms else 0.,
            'ToolR@1': ms[0]['tr'] if ms else 0.,
            'ToolP@1': ms[0]['tp'] if ms else 0.}

def recall_guard(current, baseline, tolerance=1e-12):
    """No user-unrequested recall slack. This is a validation constraint only."""
    return all(current[k] + tolerance >= baseline[k]
               for k in ['RDCR@10', 'Tool-Hit@10', 'CR-Hit@10', 'CompR@1'])

def identities(record):
    return {norm(record[k]) for k in ['query', 'original_query'] if record.get(k)}

def check_splits(train, valid, test):
    groups = [list(train), list(valid), list(test)]
    for a,b in itertools.combinations(groups,2):
        qa = set().union(*(identities(r) for r in a)) if a else set()
        qb = set().union(*(identities(r) for r in b)) if b else set()
        ia = {r['qid'] for r in a}; ib = {r['qid'] for r in b}
        if qa & qb or ia & ib:
            raise ValueError('Query/original-query/qid leakage across splits')

def build_toolsets(case, references, max_sets=32):
    """Keep original sets, then deterministic deletions/additions/replacements."""
    sets = []; inventory = set(case['inventory']['tools'])
    def add(ts):
        s = tuple(sorted(set(ts)))
        if 1 <= len(s) <= 6 and set(s) <= inventory and s not in sets:
            sets.append(s)
    for c in case['candidates']:
        add(c['tools'])
    for g in references:
        add(g['tools'])
    if len(sets) > max_sets:
        raise ValueError('Original/reference sets exceed augmentation budget')
    proposal_groups = []
    for g in references:
        G = set(g['tools'])
        if not G or not G <= inventory:
            continue
        extra = sorted(inventory-G, key=lambda t: hashlib.sha256((case['query_hash']+t).encode()).hexdigest())
        proposal_groups.append([G-{t} for t in sorted(G)])
        additions = [G|{t} for t in extra[:8]]
        additions += [G|set(extra[:n]) for n in [2,3,4] if len(extra) >= n]
        proposal_groups.append(additions)
        proposal_groups.append([(G-{t})|{e} for t,e in zip(sorted(G),extra)])
        # A partial set and the same partial set plus extras teach that coverage
        # can remain unchanged even when both candidates are imperfect.
        partial = G-{sorted(G)[-1]}
        proposal_groups.append([partial|{e} for e in extra[:4]])
    for items in itertools.zip_longest(*proposal_groups):
        for ts in items:
            if ts is not None:
                add(ts)
                if len(sets) >= max_sets:
                    return sets
    return sets

def build_pairs(candidates, references, teacher_scores, seed):
    rng = random.Random(seed); groups = {k: [] for k in PAIR_LIMITS}
    ms = [metrics(c,references) for c in candidates]
    for i,j in itertools.combinations(range(len(candidates)),2):
        a,b = candidates[i],candidates[j]; A,B = set(a['tools']),set(b['tools'])
        if a['llm'] != b['llm']:
            if A == B and signature(a) in teacher_scores and signature(b) in teacher_scores:
                delta = teacher_scores[signature(a)]-teacher_scores[signature(b)]
                if abs(delta) >= .05-1e-12:
                    groups['teacher_llm'].append((i,j) if delta > 0 else (j,i))
            continue
        if A < B or B < A:
            small,big = (i,j) if A < B else (j,i)
            S,L = set(candidates[small]['tools']),set(candidates[big]['tools'])
            # No additional hit in ANY reference. Never merge reference sets.
            same_hits = all(S & set(g['tools']) == L & set(g['tools']) for g in references)
            if same_hits and ms[small]['cp'] > ms[big]['cp']+1e-12:
                assert abs(ms[small]['cr']-ms[big]['cr']) < 1e-12
                groups['redundancy'].append((small,big))
            elif ms[big]['cr'] > ms[small]['cr']+1e-12 and ms[big]['cp'] >= ms[small]['cp']-1e-12:
                groups['missing_required'].append((big,small))
        elif len(A) == len(B):
            better,worse = (i,j) if ms[i]['cp'] > ms[j]['cp'] else (j,i)
            if ms[better]['cp'] > ms[worse]['cp']+1e-12 and ms[better]['cr'] >= ms[worse]['cr']-1e-12:
                groups['replacement'].append((better,worse))
    pairs = []; available = {}
    for kind, items in groups.items():
        available[kind] = len(items); rng.shuffle(items)
        for pos,neg in items[:PAIR_LIMITS[kind]]:
            pairs.append({'positive':pos, 'negative':neg, 'kind':kind, 'margin':.1})
    return pairs, available

def make_case(case, references, label, max_sets=32):
    llms = list(dict.fromkeys(case['inventory']['llms']))
    sets = build_toolsets(case,references,max_sets)
    candidates = [{'llm':m,'tools':list(s)} for s in sets for m in llms]
    teacher = {signature(c):label['scores'][c['id']]/100 for c in case['candidates']}
    pairs,available = build_pairs(candidates,references,teacher,case['query_hash'])
    original = {signature(c) for c in case['candidates']}
    return {'qid':case['qid'],'query':case['query'],'query_hash':case['query_hash'],
            'inventory':case['inventory'],'references':references,'candidates':candidates,
            'original_candidate_indices':[i for i,c in enumerate(candidates) if signature(c) in original],
            'pairs':pairs,'available_pair_counts':available,
            'selected_pair_counts':dict(Counter(p['kind'] for p in pairs))}
