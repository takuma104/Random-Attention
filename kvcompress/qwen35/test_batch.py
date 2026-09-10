from kvcompress.qwen35 import runtime
import torch
import pytest
from kvcompress.qwen35.cache import EvictionConfig, EvictionLayer, HybridEvictionCache, install_eviction_hooks
from kvcompress.qwen35.batch import generate_batch
from kvcompress.qwen35.generate import generate_one
from kvcompress.qwen35.test_cache import tiny_config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


def test_per_row_eviction_rng_matches_independent_sequences():
    batched=EvictionLayer(EvictionConfig(capacity=24,recent=4,seed=(13,17)),3)
    singles=[EvictionLayer(EvictionConfig(capacity=24,recent=4,seed=s),3) for s in (13,17)]
    for t in range(100):
        kv=torch.full((2,4,1,8),float(t))
        batched.update(kv,kv); batched.evict_if_needed()
        for i,s in enumerate(singles):
            s.update(kv[i:i+1],kv[i:i+1]); s.evict_if_needed()
            assert torch.equal(batched.positions[i:i+1],s.positions)
    batched.batch_select_indices(torch.tensor([1]))
    for t in range(20):
        kv=torch.full((1,4,1,8),float(t+100))
        for s in (batched,singles[1]):
            s.update(kv,kv); s.evict_if_needed()
        assert torch.equal(batched.positions,singles[1].positions)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA needed for FLA')
@torch.inference_mode()
def test_batch_model_and_b1_sampler_compatibility():
    torch.manual_seed(7)
    model=Qwen3_5ForCausalLM(tiny_config()).to('cuda',dtype=torch.bfloat16).eval()
    model.generation_config.eos_token_id=[126,127]
    handles=install_eviction_hooks(model)
    tokens=torch.randint(0,128,(2,70),device='cuda')
    tokens[1,:7]=tokens[0,:7]
    cache=HybridEvictionCache(model.config,EvictionConfig(capacity=24,recent=4,seed=(13,17)))
    singles=[HybridEvictionCache(model.config,EvictionConfig(capacity=24,recent=4,seed=s)) for s in (13,17)]
    for start,end in [(0,7)]+[(i,i+1) for i in range(7,70)]:
        a=model(tokens[:,start:end],past_key_values=cache,use_cache=True).logits
        expected=torch.cat([model(tokens[j:j+1,start:end],past_key_values=singles[j],use_cache=True).logits for j in range(2)])
        torch.testing.assert_close(a,expected,rtol=.03,atol=.03)
        for j in range(2):
            assert torch.equal(cache.layers[3].positions[j:j+1],singles[j].layers[3].positions)
    args=dict(policy='random_pp',capacity=24,recent=4,max_new_tokens=80,heartbeat=0)
    one,_=generate_one(model,tokens[:1,:7],generation_seed=19,eviction_seed=13,**args)
    batch=generate_batch(model,tokens[:1,:7],generation_seeds=[19],eviction_seeds=[13],**args)
    assert one==batch[0][0]
    for h in handles:
        h.remove()
