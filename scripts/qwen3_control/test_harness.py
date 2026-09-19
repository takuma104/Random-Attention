from copy import deepcopy
import pytest
from scripts.qwen3_control.analyze_math import check_record,seed,SAMPLING,contrasts


def record(cap):
    logical=1301; events=0 if not cap else max(0,(logical-cap)//64)
    counts=dict(kv=147456*(cap+64 if cap else logical),recurrent=0,conv=0,
        positions=2304*(cap+64) if cap else 0,queries=0)
    return dict(status='complete',grading_errors=[],batch_size=2,batch_row=0,run=0,problem_id='p',
        sampling=SAMPLING,capacity=cap,recent=64,policy='random_pp' if cap else 'native',
        prompt_tokens=2,prompt_token_ids=[20,21],generated_tokens=1300,token_ids=[10]*1299+[151645],
        eos_ids=[151645,151643],logical_cached_tokens=logical,termination='eos',final_correct=True,final_correct_terminated=True,
        generation_seed=seed('p',0,'generation'),eviction_seed=seed('p',0,'eviction'),eviction_layer_events=36*events,
        final_physical_lengths=[logical-64*events]*36 if cap else [],cache_bytes=counts,
        batch_cache_bytes_at_finish={k:2*v for k,v in counts.items()},
        elapsed_seconds=1.,batch_wall_seconds=2.,request_latency_seconds=1.9,ttft_seconds=.1)


@pytest.mark.parametrize('cap',[0,1024,2048])
def test_qwen3_record_accounting(cap):
    check_record(record(cap),cap,8192)


def test_qwen35_layer_count_and_early_eos_are_rejected():
    bad=record(1024); bad['eviction_layer_events']=8*4
    with pytest.raises(AssertionError): check_record(bad,1024,8192)
    bad=record(0); bad['token_ids'][20]=151643
    with pytest.raises(AssertionError): check_record(bad,0,8192)


def test_paired_bootstrap_keeps_two_runs_together():
    native={(p,r):dict(final_correct=True) for p in ('p','q') for r in (0,1)}
    compressed=deepcopy(native); compressed['p',0]['final_correct']=False
    rows=dict(native=native,random_pp_C1024=compressed,random_pp_C2048=deepcopy(native))
    a=contrasts(rows,['p','q']); assert a==contrasts(rows,['p','q'])
    assert a['random_pp_C1024']['difference']==-.25 and a['random_pp_C1024']['ci95']==[-.5,0.]
    assert a['random_pp_C2048']['ci95']==[0.,0.]


def test_recovery_is_frozen_shared_function_and_seed_namespace_matches():
    from scripts.qwen3_control import run_math_batch as control
    from scripts.qwen35.run_math_batch import grade_pending_group
    assert control.grade_pending_group is grade_pending_group
    assert control.stable_seed('p',1,'eviction')==seed('p',1,'eviction')
    assert control.SAMPLING.temperature==.6 and control.SAMPLING.presence_penalty==0.
