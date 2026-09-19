"""Qwen3-specific hooks; official attention/full RoPE forward is unchanged.

HybridEvictionCache is a historical class name: with Qwen3's all-full layer_types
it contains only EvictionLayer instances. The frozen selection/compaction code
and independent row/layer/head RNG are reused without modification.
"""
from kvcompress.qwen35 import runtime as _safe_import_order
from kvcompress.qwen35.cache import EvictionConfig,EvictionLayer,HybridEvictionCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,apply_rotary_pos_emb


def install_eviction_hooks(model):
    assert model.config.model_type=='qwen3'
    assert set(model.config.layer_types)=={'full_attention'}
    modules=[m for m in model.modules() if isinstance(m,Qwen3Attention)]
    assert len(modules)==model.config.num_hidden_layers
    handles=[]
    for module in modules:
        state={}
        def pre_hook(mod,args,kwargs,state=state):
            cache=kwargs.get('past_key_values')
            state['capture']=isinstance(cache,HybridEvictionCache) and cache.layers[mod.layer_idx].settings.policy=='snapkv_pp'
        def query_hook(mod,args,output,state=state):
            if state.get('capture'): state['query']=output.detach()
        def post_hook(mod,args,kwargs,output,state=state):
            cache=kwargs.get('past_key_values')
            if not isinstance(cache,HybridEvictionCache): return
            layer=cache.layers[mod.layer_idx]
            if state.get('capture'):
                q=state.pop('query').transpose(1,2)
                cos,sin=kwargs['position_embeddings']
                assert cos.shape[-1]==q.shape[-1], 'Qwen3 uses full, not partial, RoPE'
                q,_=apply_rotary_pos_emb(q,q,cos,sin)
                layer.record_queries(q)
            layer.evict_if_needed()
        handles.extend([module.register_forward_pre_hook(pre_hook,with_kwargs=True),
            module.q_norm.register_forward_hook(query_hook),module.register_forward_hook(post_hook,with_kwargs=True)])
    return handles
