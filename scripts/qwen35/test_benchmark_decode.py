import pytest
from scripts.qwen35.benchmark_decode import schedule,parse_cell


def test_fixed_schedule_and_no_eviction_capacity():
    end=schedule([8192,16384,32768],128,3,['throughput','synchronized'])
    assert end==33536
    assert parse_cell('native_preallocated',end)==('native_preallocated','random_pp',end)
    assert end<end+64  # never reaches first eviction threshold
    assert parse_cell('random_pp:1024',end)==('random_pp_C1024','random_pp',1024)


def test_reject_overlapping_or_duplicate_measurements():
    with pytest.raises(AssertionError): schedule([256,512],128,3,['throughput','synchronized'])
    with pytest.raises(AssertionError): schedule([8192],128,3,['throughput','throughput'])
    with pytest.raises(AssertionError): schedule([8192],128,0,['throughput'])
