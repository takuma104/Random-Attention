import copy
import pytest
from scripts.qwen35.analyze_frontier import bootstrap_contrast,preservation,check_prefix


def test_clustered_bootstrap_and_strict_margin():
    a={(p,r):{'score':r} for p in ('a','b','c') for r in (0,1)}
    b={(p,r):{'score':1-r} for p in ('a','b','c') for r in (0,1)}
    s=bootstrap_contrast(a,b,'score',replicates=100)
    assert s['ci95']==s['ci97_5']==[0.,0.] and s['ties']==3
    assert s['bootstrap_seed']==20260917
    assert not preservation({'ci97_5':[-.02,0.]})['supports_target']
    assert preservation({'ci97_5':[-.019,0.]})['supports_target']


def test_prefix_includes_token_sampled_after_first_compaction():
    native=dict(problem_id='p',run=0,prompt_token_ids=list(range(32)),generation_seed=1,eviction_seed=2,
                sampling={},gold='x',eos_ids=[999],token_ids=list(range(160)))
    row=copy.deepcopy(native); row.update(capacity=128,recent=16,prompt_tokens=32)
    row['token_ids'][113]=-1  # first potentially different token
    assert check_prefix(row,native)==113
    row['token_ids'][112]=-1  # first eviction forward still uses original KV
    with pytest.raises(AssertionError): check_prefix(row,native)


def test_bonferroni_interval_contains_nominal_interval():
    a={(p,r):{'score':int(p%3==0)} for p in range(30) for r in (0,1)}
    b={(p,r):{'score':int(p%5==0)} for p in range(30) for r in (0,1)}
    s=bootstrap_contrast(a,b,'score',replicates=1000)
    assert s['ci97_5'][0]<=s['ci95'][0]<=s['ci95'][1]<=s['ci97_5'][1]
