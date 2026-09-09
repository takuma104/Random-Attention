"""Full-checkpoint no-op, pre-eviction and long-position/reference correctness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen35.runtime import load_model, kernel_provenance, ROOT
import json
import time
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb, eager_attention_forward
from kvcompress.qwen35.cache import EvictionConfig, HybridEvictionCache, install_eviction_hooks


@torch.inference_mode()
def main():
    model, tokenizer, artifacts = load_model()
    # Check outputs before eviction hook compacts storage: all retained KV are
    # legal past tokens for q_len=1. An explicit no-mask eager attention is the
    # independent reference for upstream SDPA's offset-based mask construction.
    errors = []
    reference_calls = []
    def reference_hook(mod, args, kwargs, output):
        cache = kwargs.get("past_key_values")
        if not isinstance(cache, HybridEvictionCache):
            return
        layer = cache.layers[mod.layer_idx]
        if layer.events == 0 or kwargs["hidden_states"].shape[1] != 1:
            return
        x = kwargs["hidden_states"]
        shape = x.shape[:-1]
        q, gate = torch.chunk(mod.q_proj(x).view(*shape, -1, mod.head_dim*2), 2, dim=-1)
        q = mod.q_norm(q).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, *kwargs["position_embeddings"])
        a, _ = eager_attention_forward(mod, q, layer.keys, layer.values, None, scaling=mod.scaling)
        a = mod.o_proj(a.reshape(*shape, -1) * gate.reshape(*shape, -1).sigmoid())
        error = (a.float()-output[0].float()).abs().max().item()
        errors.append(error)
        # Different reduction kernels in BF16: compare bounded numerical error,
        # not exact bit identity. Log every observed maximum.
        torch.testing.assert_close(a, output[0], rtol=.03, atol=.03)
        reference_calls.append((mod.layer_idx, layer.logical_length))
    reference_handles = [m.register_forward_hook(reference_hook, with_kwargs=True)
                         for m in model.modules() if isinstance(m, Qwen3_5Attention)]
    handles = install_eviction_hooks(model)
    ids = tokenizer.apply_chat_template([{"role":"user", "content":"Compute 17 + 25."}],
                                         tokenize=True, add_generation_prompt=True, enable_thinking=True,
                                         return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
    assert ids.shape[1] < 112
    native = DynamicCache(config=model.config)
    noop = HybridEvictionCache(model.config, EvictionConfig(policy="none", capacity=128, recent=16))
    compressed = HybridEvictionCache(model.config, EvictionConfig(capacity=128, recent=16))
    current = ids
    max_noop = 0.
    max_before = 0.
    start = time.perf_counter()
    for step in range(180):
        logits = []
        for cache in (native, noop, compressed):
            logits.append(model(current, past_key_values=cache, use_cache=True, logits_to_keep=1).logits)
        a,b,c = logits
        max_noop = max(max_noop, (a.float()-b.float()).abs().max().item())
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        if compressed.get_seq_length() <= 144:
            max_before = max(max_before, (a.float()-c.float()).abs().max().item())
            torch.testing.assert_close(a,c,rtol=.02,atol=.05)
        assert torch.isfinite(c).all()
        assert compressed.get_seq_length() == native.get_seq_length()
        current = a[:, -1].argmax(-1,keepdim=True)
    assert reference_calls and compressed.eviction_events > 0
    report = dict(model_revision=artifacts["model_revision"], kernels=kernel_provenance(),
                  prompt_tokens=ids.shape[1], tested_decode_steps=179, max_noop_logit_error=max_noop,
                  max_pre_eviction_logit_error=max_before, max_reference_attention_error=max(errors),
                  reference_checks=len(reference_calls), eviction_layer_events=compressed.eviction_events,
                  elapsed_seconds=time.perf_counter()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), status="passed")
    for h in handles+reference_handles:
        h.remove()
    output = ROOT / "work/qwen35/full_model_validation.json"
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
