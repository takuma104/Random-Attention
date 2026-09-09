"""Single-sequence reproducible sampler; no model.generate private APIs.

Presence penalty follows the common serving convention: generated tokens only,
not prompt tokens. RNG for multinomial is separate from cache-selection RNG.
Top-k precedes top-p; sampling on the resulting <=20 entries avoids a full-vocab
sort. All defaults are explicitly persisted by the experiment runner.
"""
from dataclasses import dataclass, asdict
import hashlib
import time
import torch
from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin
from kvcompress.qwen35.cache import EvictionConfig, HybridEvictionCache


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = .95
    top_k: int = 20
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    min_p: float = 0.0

    def __post_init__(self):
        if self.temperature <= 0 or self.top_k < 1 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid sampling settings")
        if self.repetition_penalty != 1 or self.min_p != 0:
            raise ValueError("Only repetition_penalty=1, min_p=0 supported")


def stable_seed(problem_id, run, stream):
    payload = f"qwen35-experiment-v1:{problem_id}:{run}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63-1)


def candidate_distribution(logits, seen_generated, settings):
    scores = logits.float() - seen_generated.to(torch.float32) * settings.presence_penalty
    scores /= settings.temperature
    values, ids = scores.topk(min(settings.top_k, scores.shape[-1]), dim=-1)
    probs = values.softmax(-1)
    # Remove tokens strictly after the first token that crosses the p threshold.
    remove = probs.cumsum(-1) - probs >= settings.top_p
    values = values.masked_fill(remove, float("-inf"))
    return ids, values.softmax(-1)


def state_bytes(cache):
    kv = recurrent = conv = positions = queries = 0
    for layer in cache.layers:
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            recurrent += sum(t.numel()*t.element_size() for t in layer.recurrent_states.values() if t is not None)
            conv += sum(t.numel()*t.element_size() for t in layer.conv_states.values() if t is not None)
        else:
            for name in ("key_storage", "value_storage") if hasattr(layer, "key_storage") else ("keys", "values"):
                t = getattr(layer, name, None)
                if t is not None:
                    kv += t.numel()*t.element_size()
            for attr, kind in (("position_storage", "positions"), ("queries", "queries")):
                t = getattr(layer, attr, None)
                if t is not None:
                    if kind == "positions":
                        positions += t.numel()*t.element_size()
                    else:
                        queries += t.numel()*t.element_size()
    return dict(kv=kv, recurrent=recurrent, conv=conv, positions=positions, queries=queries)


@torch.inference_mode()
def generate_one(model, input_ids, *, policy, capacity, recent, max_new_tokens,
                 generation_seed, eviction_seed, sampling=SamplingConfig(), heartbeat=512):
    if input_ids.shape[0] != 1 or max_new_tokens < 1:
        raise ValueError("Initial runner supports batch=1 and positive output cap")
    if policy == "native":
        cache = DynamicCache(config=model.config)
    else:
        cache = HybridEvictionCache(model.config, EvictionConfig(policy, capacity, recent, eviction_seed))
    generator = torch.Generator(device=input_ids.device).manual_seed(generation_seed)
    config = model.config.get_text_config(decoder=True)
    seen = torch.zeros((1, config.vocab_size), dtype=torch.bool, device=input_ids.device)
    output = torch.empty((1, max_new_tokens), dtype=torch.long, device=input_ids.device)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos:
        raise ValueError("No EOS ids configured")
    current = input_ids
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    ttft = None
    reason = "length"
    for step in range(max_new_tokens):
        result = model(current, past_key_values=cache, use_cache=True, logits_to_keep=1)
        ids, probs = candidate_distribution(result.logits[:, -1, :], seen, sampling)
        local = torch.multinomial(probs, 1, generator=generator)
        token = ids.gather(-1, local)
        output[:, step:step+1] = token
        seen.scatter_(1, token, True)
        token_id = token.item()  # synchronizes one decode step; also checks EOS
        if step == 0:
            ttft = time.perf_counter()-start
        if token_id in eos:
            reason = "eos"
            break
        current = token
        if heartbeat and (step+1) % heartbeat == 0:
            print(f"  progress tokens={step+1} elapsed={time.perf_counter()-start:.1f}s "
                  f"layer_evictions={getattr(cache,'eviction_events',0)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    count = step+1
    metrics = dict(generated_tokens=count, termination=reason, elapsed_seconds=elapsed,
                   ttft_seconds=ttft, output_tok_s=count/elapsed,
                   decode_tok_s=(count-1)/(elapsed-ttft) if count>1 else None,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   cache_bytes=state_bytes(cache), eviction_layer_events=getattr(cache,"eviction_events",0),
                   logical_cached_tokens=cache.get_seq_length(),
                   final_physical_lengths=[layer.length for layer in cache.layers if hasattr(layer,"length")],
                   sampling=asdict(sampling), eos_ids=eos)
    # output count includes EOS; the last sampled token has not yet entered KV.
    return output[0, :count].cpu().tolist(), metrics
