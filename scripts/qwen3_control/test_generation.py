"""Exercise both REAL Qwen3 EOS IDs and stable-row residency in the shared loop."""
from kvcompress.qwen3_control.runtime import generate_batch
from transformers import Qwen3Config
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU sampler')
def test_both_qwen3_eos_ids_and_finished_row_independence():
    class ScriptedModel:
        config=Qwen3Config(vocab_size=151936,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        generation_config=SimpleNamespace(eos_token_id=[151645,151643])
        def __init__(self,scripts): self.scripts=scripts; self.step=0
        def __call__(self,current,past_key_values,**kwargs):
            for i in past_key_values.full_indices:
                layer=past_key_values.layers[i]
                x=torch.zeros(current.shape[0],1,current.shape[1],8,device=current.device,dtype=torch.bfloat16)
                layer.update(x,x); layer.evict_if_needed()
            logits=torch.full((current.shape[0],1,151936),float('-inf'),device=current.device)
            for row,script in enumerate(self.scripts): logits[row,0,script[min(self.step,len(script)-1)]]=0
            self.step+=1
            return SimpleNamespace(logits=logits)
    short=[10,11,151645]; long=[10,11,12,13,14,15,16,151643]
    ids=torch.tensor([[42],[42]],device='cuda')
    kw=dict(policy='random_pp',capacity=4,recent=1,max_new_tokens=10,heartbeat=0)
    pair=generate_batch(ScriptedModel([short,long]),ids,generation_seeds=[1,2],eviction_seeds=[3,4],**kw)
    alone=generate_batch(ScriptedModel([long]),ids[:1],generation_seeds=[2],eviction_seeds=[4],**kw)
    assert pair[0][0]==short and pair[1][0]==alone[0][0]==long
    assert [v[1]['termination'] for v in pair]==['eos','eos']
    assert [v[1]['logical_cached_tokens'] for v in pair]==[3,8]
    assert pair[0][1]['eviction_layer_events']==0
    assert pair[1][1]['eviction_layer_events']==alone[0][1]['eviction_layer_events']==8
    for _,r in pair: assert r['cache_bytes']['conv']==r['cache_bytes']['recurrent']==0
    assert pair[0][1]['request_latency_seconds']<pair[1][1]['request_latency_seconds']
    assert sum(r[1]['elapsed_seconds'] for r in pair)==pair[0][1]['batch_wall_seconds']
