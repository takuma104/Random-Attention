"""CPU tests for deterministic traces and pre-query retention accounting."""
from scripts.qwen35.run_retrieval_probe import make_case,retained,INSTRUCTION
from kvcompress.qwen35.runtime import ROOT
from kvcompress.qwen35.cache import EvictionConfig,EvictionLayer
import json
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer


def test_trace_reproducibility_and_spans():
    artifact=json.loads((ROOT/'work/qwen35/artifacts.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(artifact['model_path'],local_files_only=True)
    a=make_case(tokenizer,'calibration',0,16384)
    b=make_case(tokenizer,'calibration',0,0)
    assert a['fact_ids']==b['fact_ids'] and a['query_ids']==b['query_ids']
    assert len(a['filler_ids'])==16384 and a['key'] not in INSTRUCTION
    start,end=a['value_offsets']
    assert tokenizer.decode(a['fact_ids'][start:end]).strip()==a['options'][a['target']]
    assert a['target']==0
    c=make_case(tokenizer,'confirm',0,0)
    assert c['key']!=a['key']
    assert all(len(tokenizer.encode(x,add_special_tokens=False))==1 for x in 'ABCD')


def test_shared_selector_really_shares_heads_but_not_rows():
    layer=EvictionLayer(EvictionConfig('random_shared_pp',24,4,(1,2)),3)
    for t in range(100):
        kv=torch.full((2,4,1,8),float(t))
        layer.update(kv,kv); layer.evict_if_needed()
        assert torch.equal(layer.positions,layer.positions[:,:1].expand_as(layer.positions))
    assert not torch.equal(layer.positions[0],layer.positions[1])


def test_retention_uses_resident_positions_not_logical_age():
    # One layer, two KV heads, two rows. Only row0/head0 keeps the value at4.
    positions=torch.tensor([[[0,3,4,98],[0,3,5,98]],[[0,5,6,98],[0,5,6,98]]])
    cache=SimpleNamespace(layers=[SimpleNamespace(positions=positions)],get_seq_length=lambda:100,eviction_events=2)
    result=retained(cache,[0],3,[3,5],[4,5],2)
    assert result['fact_token_counts']==[[[2,1]],[[0,0]]]
    assert result['value_token_counts']==[[[1,0]],[[0,0]]]
    assert result['older_generated_head_jaccard_by_layer']==[[1/3],[1.]]
