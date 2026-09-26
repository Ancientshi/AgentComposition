"""Shared, target-blind Top-10 protocol and evaluation for Table 1."""
from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
import json
import re
from pathlib import Path
from collections import Counter
import baseline5_run_infer_rag_gpt as b5

ROOT = AC_ROOT

def modules():
    b4 = b5.load_module(ROOT / 'baselines/baseline4_run_infer_rag_llama.py', 'top10_b4')
    sota = b4._load_sota_module(b4._resolve_sota_script())
    return b4, sota

def prompt(row, b4, sota, tokenizer, budget=6000):
    context = row['context']['text']
    llms, tools = sota.extract_candidates_from_context(context)
    def build(evidence):
        messages = b4.build_rag_llama_messages(context=evidence, conversation_history=row['query'],
            max_tools=6, context_llms=llms, context_tools=tools)
        messages[0]['content'] += '\nRecommend exactly 10 distinct agent configurations, ordered from most to least suitable. Each configuration contains one LLM and at most 6 tools. Tool order does not make configurations distinct. Do not repeat a configuration.'
        text = messages[1]['content']
        text = text.replace('Select one retrieved LLM and at most 6 necessary retrieved tools',
                            'For each of 10 distinct configurations, select one retrieved LLM and at most 6 necessary retrieved tools')
        text = text.replace('Return exactly one line.', 'Return exactly 10 lines, one configuration per line, in descending recommendation order. No numbering.')
        text = text.replace('6. Stop immediately after <SPECIAL_END>.', '6. After <SPECIAL_END>, start the next configuration on a new line. Stop after the tenth configuration.')
        text = text.replace('Output the resource-selection line only.', 'Output the 10 resource-selection lines only.')
        messages[1]['content'] = text
        return messages
    messages = build(context)
    ids = tokenizer.encode(context, add_special_tokens=False)
    while len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)) > budget:
        if not ids:
            raise ValueError('Inventories and full query exceed input budget; refusing to truncate query')
        ids = ids[:max(0, len(ids)-256)]
        messages = build(tokenizer.decode(ids, skip_special_tokens=True))
    return messages, llms, tools, {'messages_sha256': b5.digest(messages),
        'reference_context_sha256': b5.digest(context), 'evidence_truncated': len(ids) < len(tokenizer.encode(context, add_special_tokens=False)),
        'input_token_count': len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)),
        'full_query_preserved': True}

def parse_candidates(text, llms, tools, b4, sota, previous=()):
    text = re.sub(r'<+\s*(SPECIAL_END|TOOL_SEP|TOOL_EMPTY)\s*>+', r'<\1>', text)
    results = list(previous)
    seen = {(r['llm_token'], tuple(sorted(set(r['tool_tokens'])))) for r in results}
    rejected = Counter()
    # Each LLM token starts a configuration. Some models omit END between
    # otherwise complete lines; never concatenate their tools into one agent.
    for chunk in re.split(r'(?=<+\s*LLM_[^<>\n]+>)', text):
        start = chunk.find('<LLM_')
        if start < 0:
            continue
        candidate = chunk[start:].split('<SPECIAL_END>',1)[0].strip()
        end_present = '<SPECIAL_END>' in chunk
        if '<TOOL_SEP>' in candidate:
            body = candidate.split('<TOOL_SEP>',1)[1]
            remainder = re.sub(r'<+[^<>\n]+>+', '', body).strip(' \t\n\r,;')
            if remainder:
                rejected['incomplete_or_non_token_tool_body'] += 1
                continue
        candidate += ' <SPECIAL_END>'
        llm, selected, normalization = b4.parse_structured_tokens_relaxed_protocol(candidate,
            context_llms=llms, context_tools=tools, tool_sep_token='<TOOL_SEP>',
            end_token='<SPECIAL_END>', tool_empty_token='<TOOL_EMPTY>')
        selected = [t for t in selected if t != '<TOOL_EMPTY>']
        if not llm or len(set(selected)) > 6:
            rejected['invalid_llm_or_size'] += 1
            continue
        if '<TOOL_SEP>' not in candidate or (not selected and '<TOOL_EMPTY>' not in candidate):
            rejected['invalid_structure'] += 1
            continue
        key = (llm, tuple(sorted(set(selected))))
        if key in seen:
            rejected['duplicate'] += 1
            continue
        seen.add(key)
        strict = sota.build_strict_text(llm_token=llm, tool_tokens=selected,
            tool_sep_token='<TOOL_SEP>', end_token='<SPECIAL_END>', tool_empty_token='<TOOL_EMPTY>')
        results.append({'rank': len(results)+1, 'llm_token': llm, 'tool_tokens': selected,
                        'strict_text': strict, 'gen_text': strict, 'protocol_normalization': normalization,
                        'end_marker_present': end_present,
                        'inventory_valid': llm in llms and all(t in tools for t in selected)})
        if len(results) == 10:
            break
    return results, dict(rejected)

def evaluate(output, records, expected):
    if len(records) != expected:
        raise ValueError(f'Incomplete experiment: {len(records)}/{expected}')
    ev = b5.load_module(ROOT / 'evaluation/reference/exp3/evaluate_ranked_recall_baseline2.py', 'top10_evaluator')
    rows = [ev.evaluate_one(r) for r in records]
    for row in rows:
        row['cr_hit@1'] = row['top1_complete_recall']
    keys = ['top1_tool_recall', 'tool_hit@10', 'top1_component_recall',
            'cr_hit@1', 'cr_hit@10', 'cr_mrr@10', 'rdcr@10',
            'top1_tool_precision', 'top1_tool_f1']
    mean = {k: sum(r[k] for r in rows)/len(rows) for k in keys}
    report = {'n': len(rows), 'mean': mean, 'candidate_counts': dict(Counter(len(r['results']) for r in records)),
              'gold_source': 'dataset_example.target only', 'missing_rank_policy': 'zero',
              'all_have_ten': all(len(r['results']) == 10 for r in records)}
    b5.write_json(output/'evaluation/top10_metrics.json', report)
    values = [f'{mean[k]*100:.2f}' if k != 'cr_mrr@10' else f'{mean[k]:.4f}' for k in keys[:7]]
    (output/'evaluation/table1_row.tex').write_text(records[0]['baseline']+' & '+' & '.join(values)+r' \\'+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report
