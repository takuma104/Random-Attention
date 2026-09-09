from kvcompress.qwen35 import runtime  # initialize FLA before Transformers modeling
import pytest
import torch
from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from kvcompress.qwen35.cache import EvictionConfig, EvictionLayer, HybridEvictionCache, install_eviction_hooks


def test_config_and_prompt_overflow():
    with pytest.raises(ValueError):
        EvictionConfig(capacity=4, recent=4)
    layer = EvictionLayer(EvictionConfig(capacity=16, recent=4), 3)
    with pytest.raises(ValueError, match="Prompt"):
        layer.update(torch.zeros(1, 2, 12, 8), torch.zeros(1, 2, 12, 8))


@pytest.mark.parametrize("policy", ["random_pp", "random_shared_pp", "recency_pp", "random", "snapkv_pp"])
def test_eviction_invariants(policy):
    cfg = EvictionConfig(policy=policy, capacity=24, recent=4, seed=13)
    layer = EvictionLayer(cfg, 3)
    duplicate = EvictionLayer(cfg, 3)
    for t in range(100):
        k = torch.full((2, 4, 1, 8), float(t))
        v = k + 1000
        for target in (layer, duplicate):
            target.update(k, v)
            if policy == "snapkv_pp":
                target.record_queries(torch.ones(2, 8, 1, 8))
            target.evict_if_needed()
        assert torch.equal(layer.positions, duplicate.positions)
        assert layer.get_seq_length() == t+1
        assert layer.length <= 28
        assert layer.key_storage.numel() == 2*4*28*8
        assert torch.equal(layer.keys[..., 0].long(), layer.positions)
        assert torch.equal(layer.values[..., 0].long(), layer.positions + 1000)
        assert bool((layer.positions.diff(dim=-1) > 0).all())
        if policy.endswith("_pp"):
            assert bool((layer.positions[..., 0] == 0).all())
        tail_len = min(t+1, cfg.recent)
        assert torch.equal(layer.positions[0, 0, -tail_len:], torch.arange(t+1-tail_len, t+1))
        if policy == "random_shared_pp":
            assert torch.equal(layer.positions[:, :1].expand_as(layer.positions), layer.positions)
    assert layer.events > 10
    if policy == "random_pp":
        assert not torch.equal(layer.positions[0, 0], layer.positions[0, 1])
    saved = layer.positions[1:2].clone()
    layer.batch_select_indices(torch.tensor([1]))
    assert torch.equal(layer.positions, saved)


def tiny_config():
    return Qwen3_5TextConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention"]*3+["full_attention"]+["linear_attention"]*3+["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .5,
                         "mrope_section": [3, 3, 2]},
        attn_implementation="sdpa",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FLA")
@torch.inference_mode()
def test_tiny_hybrid_native_equivalence_and_eviction():
    torch.manual_seed(4)
    model = Qwen3_5ForCausalLM(tiny_config()).to(device="cuda", dtype=torch.bfloat16).eval()
    handles = install_eviction_hooks(model)
    native = DynamicCache(config=model.config)
    noop = HybridEvictionCache(model.config, EvictionConfig(policy="none", capacity=24, recent=4))
    bounded = HybridEvictionCache(model.config, EvictionConfig(capacity=24, recent=4))
    tokens = torch.randint(0, 128, (1, 48), device="cuda")
    # No-op adapter uses upstream storage and logits must be bit-identical.
    for start, end in [(0, 7)] + [(i, i+1) for i in range(7, 48)]:
        x = tokens[:, start:end]
        a = model(x, past_key_values=native, use_cache=True).logits
        b = model(x, past_key_values=noop, use_cache=True).logits
        c = model(x, past_key_values=bounded, use_cache=True).logits
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        if end <= 28:
            torch.testing.assert_close(a, c, rtol=.02, atol=.02)
        assert torch.isfinite(c).all()
        assert bounded.get_seq_length() == end
    assert bounded.eviction_events > 0
    assert bounded.layers[3].length <= 28
    assert bounded.layers[3].get_mask_sizes(1) == (bounded.layers[3].length+1, 48-bounded.layers[3].length)
    # Eviction operations must never touch DeltaNet state (even when evict fires).
    snapshots = {}
    for i, layer in enumerate(bounded.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            snapshots[i] = ([x.clone() for x in layer.conv_states.values()], [x.clone() for x in layer.recurrent_states.values()])
    for i in bounded.full_indices:
        layer = bounded.layers[i]
        while layer.length < 28:
            layer.update(torch.zeros(1, 2, 1, 32, device="cuda", dtype=torch.bfloat16),
                         torch.zeros(1, 2, 1, 32, device="cuda", dtype=torch.bfloat16))
        assert layer.evict_if_needed()
    for i, (convs, states) in snapshots.items():
        for actual, expected in zip(bounded.layers[i].conv_states.values(), convs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(bounded.layers[i].recurrent_states.values(), states):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for h in handles:
        h.remove()
