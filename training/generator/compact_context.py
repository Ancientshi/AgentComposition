"""Target-blind context compression shared by training and inference."""
from __future__ import annotations
import hashlib
import html
import re

VERSION = 'compact-context-v1'
TASK = 'Select the best target agent from the provided context. Output only the target agent.'
TOKEN_RE = re.compile(r'<<[^<>\n]+>>|<LLM_[^<>\n]+>|<TOOL_[^<>\n]+>')
SPECIAL = {'<TOOL_SEP>', '<TOOL_EMPTY>', '<SPECIAL_END>'}


def query_key(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def query_hash(text):
    return hashlib.sha256(query_key(text).encode()).hexdigest()


def components(text):
    tokens = list(dict.fromkeys(t for t in TOKEN_RE.findall(text or '') if t not in SPECIAL))
    return [t for t in tokens if t.startswith('<LLM_')], [t for t in tokens if not t.startswith('<LLM_')]


def shorten(text, tokenizer, limit):
    if limit <= 0:
        return ''
    text = html.unescape(re.sub(r'<[^>]+>', ' ', text or ''))
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if 'gold' in text.lower() and any(w in text.lower() for w in ['inject', 'supervised', 'target']):
        return ''
    # Endpoint semantics are more useful than repeated API marketing descriptions.
    if 'Api Description:' in text:
        text = text.rsplit('Api Description:', 1)[-1].strip()
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:limit], skip_special_tokens=False).strip()


def parse_context(context):
    llms, tools = components(context)
    descriptions, strengths, scores = {}, {}, {}
    bundles = []
    section = ''
    for line in context.splitlines():
        if line.strip().endswith(':') and not line.startswith('  '):
            section = line.strip()[:-1]
        if section == 'cf_retrieved_tool_bundle':
            # Match closing brace followed by evidence reference; tool names can contain {id}.
            for m in re.finditer(r'\{(.*?)\}\s*\[\d+\]', line):
                ts = components(m.group(1))[1]
                if ts and ts not in bundles:
                    bundles.append(ts)
        m = re.search(r'\btoken=(<<[^<>]+>>|<[^<>]+>)', line)
        if m:
            token = m.group(1)
            d = re.search(r'\bdesc=(.*)$', line)
            st = re.search(r'\bstrengths=(.*?)(?:\s+\||$)', line)
            sc = re.search(r'\b(?:norm_score|raw_score|score)=(-?[\d.]+)', line)
            if d and token not in descriptions:
                descriptions[token] = d.group(1)
            if st and token not in strengths:
                strengths[token] = st.group(1)
            if sc and token not in scores:
                scores[token] = sc.group(1)
        if '| tools=' in line:
            segment = line.split('| tools=', 1)[1]
            matches = list(TOKEN_RE.finditer(segment))
            for i, tm in enumerate(matches):
                token = tm.group(0)
                end = matches[i + 1].start() if i + 1 < len(matches) else len(segment)
                descriptions.setdefault(token, segment[tm.end():end].strip(' :;'))
    return llms, tools, bundles, descriptions, strengths, scores


def render(parsed, tokenizer, tool_desc_tokens=32, llm_desc_tokens=48):
    llms, tools, bundles, descriptions, strengths, scores = parsed
    lines = ['LLM candidates (retrieval order):']
    for i, token in enumerate(llms, 1):
        info = shorten(strengths.get(token) or descriptions.get(token, ''), tokenizer, llm_desc_tokens)
        lines.append(f'{i}. {token}' + (f' | {info}' if info else ''))
    lines += ['', 'Retrieved tool bundles:']
    lines += [f'{i}. ' + ' '.join(ts) for i, ts in enumerate(bundles, 1)] or ['(none)']
    lines += ['', 'Tool candidates (retrieval order):']
    for i, token in enumerate(tools, 1):
        info = shorten(descriptions.get(token, ''), tokenizer, tool_desc_tokens)
        lines.append(f'{i}. {token}' + (f' | {info}' if info else ''))
    return '\n'.join(lines)


def prompt_text(context, query):
    return f'### Task:\n{TASK}\n\n### Context:\n{context}\n\n### User Query:\n{query.strip()}\n\n### Answer:\n'


def build_prompt(context, query, tokenizer, max_prompt_tokens=6144):
    """Never inspect targets or remove candidates/query to meet a token budget."""
    parsed = parse_context(context)
    for tool_n, llm_n in [(32,48),(24,40),(16,32),(8,16),(0,0)]:
        compact = render(parsed, tokenizer, tool_n, llm_n)
        prompt = prompt_text(compact, query)
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) <= max_prompt_tokens:
            # Validate exact candidate identities even after HTML/description cleaning.
            old = set(parsed[0] + parsed[1])
            new = set(sum(components(compact), []))
            if not old <= new:
                raise ValueError('Compression lost a candidate identity')
            return prompt, {'version':VERSION, 'prompt_tokens':len(ids), 'tool_description_tokens':tool_n,
                            'llm_description_tokens':llm_n, 'llm_candidates':len(parsed[0]),
                            'tool_candidates':len(parsed[1]), 'bundles':len(parsed[2])}
    raise ValueError('Query and candidate names exceed prompt budget even without descriptions')


def encode_supervision(prompt, target, tokenizer, max_seq_len=8192):
    """Explicit full-target completion mask; reject instead of silently truncating."""
    if not target.startswith('<LLM_') or not target.endswith('<SPECIAL_END>'):
        raise ValueError('Expected complete LLM + tools target')
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    full_ids = tokenizer.encode(prompt + target + tokenizer.eos_token, add_special_tokens=True)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Tokenizer boundary mismatch at answer start')
    if len(full_ids) > max_seq_len:
        raise ValueError('Complete target would exceed max_seq_len')
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if not any(t != -100 for t in labels):
        raise ValueError('No supervised answer tokens')
    return {'input_ids':full_ids, 'labels':labels, 'attention_mask':[1]*len(full_ids)}
