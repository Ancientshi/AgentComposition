from agentcomposition.paths import ROOT as AC_ROOT, BASE_MODEL as AC_BASE_MODEL, EASYREC_MODEL as AC_EASYREC_MODEL, env as AC_ENV
from pathlib import Path
import hashlib,json
R=Path(str(AC_ROOT));W=R/'experiments/domain_generalization/sft';p=R/'inference/run_infer_v13_batch_eval_sota.py';s=p.read_text()
old='''    critic.score_nodes(query=query, nodes=completed, evidence_context=context, stage="final_rerank")
    assign_search_scores(
        completed,
        mode=search_score_mode,
        critic_weight=critic_weight,
        generator_weight=generator_weight,
        length_penalty=length_penalty,
    )
    completed.sort(
        key=lambda n: (
            float(n.search_score if n.search_score is not None else -1e30),
            float(n.critic_raw if n.critic_raw is not None else -1e30),
            n.generator_avg_logprob,
        ),
        reverse=True,
    )'''
new='''    assign_generator_only_scores(completed, length_penalty=length_penalty)
    completed.sort(key=lambda n: (-float(n.search_score), -n.generator_avg_logprob, int(n.node_id[1:])))'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('"final_critic_reranking": True','"final_critic_reranking": False').replace('"critic_policy": "final_rerank_only"','"critic_policy": "disabled"').replace('component_level_generator_guided_beam_final_critic_rerank','component_level_generator_only_beam').replace('"generator_guided_beam_final_critic_rerank"','"generator_only_beam"')
s=s.replace('final critic rerank: scoring all','final generator-only ranking: scoring all')
(W/'sota_no_critic.py').write_text(s)
(W/'inference_patch.json').write_text(json.dumps({'original_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'patched_sha256':hashlib.sha256(s.encode()).hexdigest(),'changes':'replace only final critic call/ranking with generator_avg_logprob - length_penalty*size, chronological ties; update metadata; generation search unchanged'},indent=2))
