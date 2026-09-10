"""CPU-only tests: safe to run while the frozen inference job is active."""
import pytest
from scripts.qwen35.analyze_main import holm_adjust,primary_decisions,SELECTORS
from scripts.qwen35.analyze_math import paired_stats


def test_holm_monotone_and_capped():
    assert holm_adjust({'a':.04,'b':.041})=={'a':.08,'b':.08}
    assert holm_adjust({'a':.01,'b':.2})=={'a':.02,'b':.2}
    assert holm_adjust({'a':.8,'b':.9})=={'a':1.,'b':1.}
    with pytest.raises(ValueError):
        holm_adjust({'a':float('nan')})


def test_preservation_strict_margin_and_selector_gate():
    def stats(ci,p=.01):
        return {'final_correct':dict(ci95=ci,exact_sign_p=p)}
    contrasts={'random_pp_C2048 - native':stats([-.02,.01]),
               SELECTORS[0]:stats([.01,.1]),SELECTORS[1]:stats([-.01,.02],.2)}
    result=primary_decisions(contrasts)
    assert not result['accuracy_preservation']['supports_target']
    assert result['selector_comparisons'][SELECTORS[0]]['supported_direction']=='random'
    assert result['selector_comparisons'][SELECTORS[1]]['supported_direction']=='unresolved'
    contrasts['random_pp_C2048 - native']=stats([-.019,.01])
    contrasts[SELECTORS[0]]=stats([-.1,-.01])
    result=primary_decisions(contrasts)
    assert result['accuracy_preservation']['supports_target']
    assert result['selector_comparisons'][SELECTORS[0]]['supported_direction']=='comparator'


def test_cluster_keeps_both_runs_together():
    a={(p,r):{'score':r} for p in ['a','b','c'] for r in [0,1]}
    b={(p,r):{'score':1-r} for p in ['a','b','c'] for r in [0,1]}
    # Every problem has mean difference zero, despite opposite per-run outcomes.
    stats=paired_stats(a,b,'score',replicates=100)
    assert stats['ci95']==[0.,0.]
    assert stats['problem_ties']==3 and stats['exact_sign_p']==1.
