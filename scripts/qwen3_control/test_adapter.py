import pytest
from kvcompress.qwen3_control import runtime  # preserve shared import order
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,apply_rotary_pos_emb,eager_attention_forward
from kvcompress.qwen3_control.cache import EvictionConfig,HybridEvictionCache,install_eviction_hooks
from kvcompress.qwen35.generate import state_bytes


@pytest.mark.parametrize('policy',['random_pp','recency_pp','snapkv_pp'])
@torch.inference_mode()
def test_official_qwen3_forward_with_shared_cache(policy):
    old_threads=torch.get_num_threads(); torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    cfg._attn_implementation='sdpa'; model=Qwen3ForCausalLM(cfg).eval()
    checks=[]
    def reference(mod,args,kwargs,output):
        cache=kwargs.get('past_key_values')
        if not isinstance(cache,HybridEvictionCache) or cache.layers[mod.layer_idx].events==0: return
        x=kwargs['hidden_states']; q=mod.q_norm(mod.q_proj(x).view(*x.shape[:-1],-1,mod.head_dim)).transpose(1,2)
        q,_=apply_rotary_pos_emb(q,q,*kwargs['position_embeddings'])
        layer=cache.layers[mod.layer_idx]
        expected,_=eager_attention_forward(mod,q,layer.keys,layer.values,None,scaling=mod.scaling)
        expected=mod.o_proj(expected.reshape(*x.shape[:-1],-1))
        torch.testing.assert_close(expected,output[0],rtol=1e-5,atol=1e-6)
        checks.append(mod.layer_idx)
    refs=[m.register_forward_hook(reference,with_kwargs=True) for m in model.modules() if isinstance(m,Qwen3Attention)]
    handles=install_eviction_hooks(model)
    native=DynamicCache(config=cfg)
    noop=HybridEvictionCache(cfg,EvictionConfig('none',16,4,(1,2)))
    compressed=HybridEvictionCache(cfg,EvictionConfig(policy,16,4,(1,2)))
    current=torch.tensor([[3,4,5],[3,4,5]])
    for step in range(40):
        a=model(current,past_key_values=native,use_cache=True).logits
        b=model(current,past_key_values=noop,use_cache=True).logits
        c=model(current,past_key_values=compressed,use_cache=True).logits
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        if compressed.get_seq_length()<=20: torch.testing.assert_close(a,c,rtol=0,atol=0)
        current=torch.full((2,1),6+step%11)
    assert compressed.full_indices==[0,1]
    assert compressed.eviction_events==2*((42-16)//4) and checks
    for layer in compressed.layers:
        assert torch.equal(layer.positions[...,:3],torch.arange(3).expand(2,1,3))
        assert torch.all(layer.positions[...,1:]>layer.positions[...,:-1])
        assert layer.length<=20
        if policy=='snapkv_pp': assert layer.queries.shape==(2,2,4,8)
    counts=state_bytes(compressed); assert counts['recurrent']==counts['conv']==0
    for h in refs+handles: h.remove()
    torch.set_num_threads(old_threads)


def test_control_sampling_is_explicit_and_not_qwen35_default():
    from kvcompress.qwen35.generate import SamplingConfig
    assert runtime.SAMPLING.temperature==.6 and runtime.SAMPLING.presence_penalty==0.
    assert SamplingConfig().temperature==1. and SamplingConfig().presence_penalty==1.5
