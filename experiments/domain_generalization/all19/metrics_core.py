"""Frozen metric functions copied verbatim from the prior eight-metric evaluator."""
import math

def evaluate(candidates, gold):
    """Use one reference at a time; maximize each metric independently, as recall does."""
    resolved = [g for g in gold if g['llm']]
    assert resolved
    per_rank = []
    for candidate in (candidates[:10] + [None] * 10)[:10]:
        if candidate is None:
            per_rank.append(dict.fromkeys(['tr', 'cr', 'tp', 'cp', 'tf1', 'cf1', 'th', 'ch'], 0.0))
            continue
        tools = set(candidate['tools'])
        def overlap(g): return len(tools & set(g['tools']))
        def match(g): return int(bool(g['llm']) and candidate['llm'] == g['llm'])
        per_rank.append({
            'tr': max(overlap(g) / len(set(g['tools'])) if g['tools'] else 1.0 for g in gold),
            'cr': max((overlap(g) + match(g)) / (1 + len(set(g['tools']))) for g in resolved),
            'tp': max(overlap(g) / len(tools) if tools else float(not g['tools']) for g in gold),
            'cp': max((overlap(g) + match(g)) / (1 + len(tools)) for g in resolved),
            'tf1': max(2 * overlap(g) / (len(tools) + len(set(g['tools'])))
                       if tools or g['tools'] else 1.0 for g in gold),
            'cf1': max(2 * (overlap(g) + match(g)) / (2 + len(tools) + len(set(g['tools']))) for g in resolved),
            'th': float(any(set(g['tools']) <= tools for g in gold)),
            'ch': float(any(match(g) and set(g['tools']) <= tools for g in resolved)),
        })
    weights = [1 / math.log2(r + 1) for r in range(1, 11)]
    def discounted(key): return sum(w * p[key] for w, p in zip(weights, per_rank)) / sum(weights)
    ranks = [i + 1 for i, p in enumerate(per_rank) if p['ch']]
    first = per_rank[0]
    return {'ToolR@1': first['tr'], 'Tool-Hit@10': max(p['th'] for p in per_rank),
            'CompR@1': first['cr'], 'CR-Hit@1': first['ch'], 'CR-Hit@10': float(bool(ranks)),
            'CR-MRR@10': 1 / ranks[0] if ranks else 0.0, 'RDCR@10': discounted('cr'),
            'ToolP@1': first['tp'], 'CompP@1': first['cp'], 'ToolF1@1': first['tf1'],
            'CompF1@1': first['cf1'], 'RDCP@10': discounted('cp'),
            'RDCF1@10': discounted('cf1')}

def skills_candidates(raw, component_map, llm_map):
    """Literal identifier protocol: keep invalid/duplicate/missing slots at their ranks."""
    predicted, seen = [], set()
    for c in (raw.get('results') or [])[:10]:
        try:
            assert isinstance(c, dict)
            llm = c.get('llm', '')
            valid_llm = llm if isinstance(llm, str) and llm in llm_map else None
            ts = c['tools']
            assert isinstance(ts, list) and 1 <= len(ts) <= 10
            assert all(isinstance(t, str) and t in component_map for t in ts)
            assert len(set(ts)) == len(ts)
            key = (str(llm) if valid_llm is None else valid_llm, tuple(sorted(ts)))
            assert key not in seen
            seen.add(key)
            predicted.append({'llm': llm_map.get(valid_llm), 'tools': [component_map[t] for t in ts]})
        except (AssertionError, KeyError, TypeError):
            predicted.append(None)
    return predicted + [None] * (10 - len(predicted))
