import copy
import pytest
from scripts.qwen35.analyze_efficiency import audit_window,distribution,MIB


def test_window_rejects_wrong_position_or_cache_storage():
    kwargs=dict(name='native_dynamic',batch=2,capacity=None,contexts=[8192],steps=128,
                repeats=3,modes=['throughput','synchronized'],maximum=8960)
    row=dict(method='native_dynamic',batch=2,steps=128,nominal_context=8192,repeat=0,mode='throughput',
        logical_start=8192,logical_end=8320,elapsed_seconds=2.,ms_per_step=15.625,tokens_per_second=128.,
        eviction_layer_events=0,cache_bytes=dict(kv=2*32768*8320,recurrent=96*MIB,conv=3*MIB,positions=0,queries=0),
        peak_allocated_bytes=12000000000,peak_reserved_bytes=13000000000)
    assert audit_window(row,**kwargs)==(8192,0,'throughput')
    wrong=copy.deepcopy(row); wrong['logical_start']+=1
    with pytest.raises(AssertionError): audit_window(wrong,**kwargs)
    wrong=copy.deepcopy(row); wrong['cache_bytes']['kv']+=1
    with pytest.raises(AssertionError): audit_window(wrong,**kwargs)


def test_descriptive_median_is_not_a_confidence_interval():
    assert distribution([1.,3.,2.])==dict(median=2.,minimum=1.,maximum=3.,n=3)
    with pytest.raises(AssertionError): distribution([float('nan')])
