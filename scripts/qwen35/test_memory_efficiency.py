import copy
import pytest
from scripts.qwen35.analyze_memory_efficiency import choose_batch,paired_metrics


def test_selection_uses_memory_and_discloses_grid_fallback():
    p=dict(batch=96,peak_allocated_gib=97.6,tokens_per_second=-100)
    assert choose_batch([p],100)==(p,True)
    low=dict(batch=64,peak_allocated_gib=94.787)
    high=dict(batch=72,peak_allocated_gib=100.562)
    assert choose_batch([low,high],100)==(low,False)
    with pytest.raises(AssertionError): choose_batch([low],100)


def test_pairing_preserves_work_per_sequence_not_batch():
    rows=[dict(nominal_context=32768,repeat=0,mode=mode,logical_start=start,logical_end=start+128,
        steps=128,tokens_per_second=100.,ms_per_step=80.,batch=8)
        for mode,start in [('throughput',32768),('synchronized',32896)]]
    large=copy.deepcopy(rows)
    for r in large: r.update(batch=96,tokens_per_second=600.,ms_per_step=160.)
    result=paired_metrics(large,rows)
    assert result[0]['paired_aggregate_throughput_ratio']['median']==6.
    assert result[0]['paired_step_latency_ratio']['median']==2.
    large[0]['logical_start']+=1
    with pytest.raises(AssertionError): paired_metrics(large,rows)
