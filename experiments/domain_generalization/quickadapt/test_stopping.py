from stopping import stop_reason
h=lambda x:[{'step':i*2,'loss':v} for i,v in enumerate(x)]
assert stop_reason(h([1,.9,.8,.7,.6,.5])) is None
assert stop_reason(h([.2,.2001,.1999,.2,.2002,.2]))=='five_checks_stable'
assert stop_reason(h([.3,.2,.2001,.2,.1999,.2])) is None
assert stop_reason(h([.2,.21,.211,.212]))=='validation_deterioration'
assert stop_reason(h([.2,.2,.2,.2,.2])) is None
assert stop_reason(h([.2,float('nan')]))=='nonfinite'
print('PASS: plateau, ongoing improvement, recent improvement, deterioration, minimum checks, nonfinite')
