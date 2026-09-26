"""Pure validation-only stopping rule; step zero can win checkpoint selection."""
import math

def stop_reason(history):
 if any(not math.isfinite(x['loss']) for x in history):return 'nonfinite'
 # Require three post-update evaluations, each >3% worse than its preceding best.
 if len(history)>=4:
  last=history[-3:];start=len(history)-3
  if all(v['loss']>1.03*min(x['loss'] for x in history[:start+i]) for i,v in enumerate(last)):return 'validation_deterioration'
 # Five consecutive post-update checks, spanning eight parameter updates.
 if len(history)>=6:
  window=history[-5:];ls=[x['loss'] for x in window];mean=sum(ls)/5
  prior_best=min(x['loss'] for x in history[:-5]);relative_gain=(prior_best-min(prior_best,min(ls)))/max(prior_best,1e-12)
  if (max(ls)-min(ls))/max(mean,1e-12)<=.01 and relative_gain<=.01:return 'five_checks_stable'
 return None
