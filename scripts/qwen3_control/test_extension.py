from copy import deepcopy
import pytest
from scripts.qwen3_control.audit_extension import check_prefix


def record():
    return dict(problem_id='p',run=0,prompt_token_ids=[1],generation_seed=1,eviction_seed=2,
        policy='native',capacity=0,recent=64,sampling={},eos_ids=[151645,151643],token_ids=[10,11],termination='length')


def test_cap_extension():
    a=record(); b=deepcopy(a); b['token_ids']+=[151645]; b['termination']='eos'
    assert check_prefix(a,b)==2


def test_eos_cannot_extend():
    a=record(); a['token_ids'][-1]=151645; a['termination']='eos'
    b=deepcopy(a); b['token_ids']+=[10]
    with pytest.raises(AssertionError): check_prefix(a,b)


def test_token_drift_rejected():
    a=record(); b=deepcopy(a); b['token_ids'][0]=12
    with pytest.raises(AssertionError): check_prefix(a,b)
