"""Full-Attention-only eviction; leaves the upstream Qwen3.5 forward intact.

Supported workload: an unpadded prefill followed by single-token decode. A
batch contains equal-length prompts; no chunked continuation/speculation.
C includes the protected recent r slots (unlike the paper's textual K).
Full KV storage is preallocated to C+r, positions stay absolute, and eviction
runs in an attention post-hook (after the current attention has consumed KV).
DeltaNet cache layers are created and updated exclusively by Transformers.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, apply_rotary_pos_emb


@dataclass(frozen=True)
class EvictionConfig:
    policy: str = "random_pp"
    capacity: int = 1024
    recent: int = 64
    seed: int | tuple[int, ...] = 0

    def __post_init__(self):
        if self.policy not in {"none", "random_pp", "random", "recency_pp", "random_shared_pp", "snapkv_pp"}:
            raise ValueError(self.policy)
        if self.recent < 1 or self.capacity <= self.recent:
            raise ValueError("Require capacity > recent >= 1")


class EvictionLayer(DynamicLayer):
    def __init__(self, settings: EvictionConfig, layer_idx: int):
        super().__init__()
        self.settings = settings
        self.layer_idx = layer_idx
        self.logical_length = 0
        self.length = 0
        self.prompt_length = 0
        self.events = 0
        self.queries = None
        self.generator = None

    def update(self, key_states, value_states, *args, **kwargs):
        n = key_states.shape[-2]
        if self.logical_length and n != 1:
            raise ValueError("Only single-token decode after initial prefill is supported")
        if not self.logical_length:
            self.prompt_length = n
            if self.settings.policy != "none" and n >= self.settings.capacity - self.settings.recent:
                raise ValueError(f"Prompt {n} does not fit C-r={self.settings.capacity-self.settings.recent}")
        if self.settings.policy == "none":
            result = super().update(key_states, value_states)
            self.logical_length += n
            self.length += n
            return result
        if not self.is_initialized:
            self.dtype, self.device = key_states.dtype, key_states.device
            b, h, _, d = key_states.shape
            self.key_storage = torch.empty(b, h, self.settings.capacity + self.settings.recent, d,
                                           device=self.device, dtype=self.dtype)
            self.value_storage = torch.empty_like(self.key_storage)
            self.position_storage = torch.empty(b, h, self.settings.capacity + self.settings.recent,
                                                device=self.device, dtype=torch.long)
            seeds = self.settings.seed
            if isinstance(seeds, tuple):
                if len(seeds) != b:
                    raise ValueError('One eviction seed per batch row is required')
                self.generators = [torch.Generator(device=self.device).manual_seed(s + 100003*self.layer_idx) for s in seeds]
            else:
                self.generator = torch.Generator(device=self.device).manual_seed(seeds + 100003*self.layer_idx)
            self.is_initialized = True
        end = self.length + n
        if end > self.key_storage.shape[-2]:
            raise RuntimeError("KV overflow: attention eviction hook missing?")
        self.key_storage[..., self.length:end, :].copy_(key_states)
        self.value_storage[..., self.length:end, :].copy_(value_states)
        self.position_storage[..., self.length:end] = torch.arange(self.logical_length, self.logical_length + n,
                                                                    device=self.device)
        self.length = end
        self.logical_length += n
        self._refresh_views()
        return self.keys, self.values

    def _refresh_views(self):
        self.keys = self.key_storage[..., :self.length, :]
        self.values = self.value_storage[..., :self.length, :]
        self.positions = self.position_storage[..., :self.length]

    def get_seq_length(self):
        return self.logical_length

    def get_mask_sizes(self, query_length):
        # The offset aligns the final physical slot with the current logical query.
        # Interior positions differ per head, but for single-token decode every
        # retained token is in the past, so no per-head causal mask is needed.
        return self.length + query_length, self.logical_length - self.length

    def record_queries(self, queries):
        self.queries = queries[..., -self.settings.recent:, :].detach() if self.queries is None else torch.cat(
            (self.queries, queries), dim=-2)[..., -self.settings.recent:, :].contiguous()

    def evict_if_needed(self):
        cfg = self.settings
        if cfg.policy == "none" or self.length < cfg.capacity + cfg.recent:
            return False
        b, h, n, d = self.keys.shape
        candidates = n - cfg.recent
        keep_n = cfg.capacity - cfg.recent
        positions = self.positions[..., :candidates]
        if cfg.policy == "recency_pp":
            scores = positions.float()
        elif cfg.policy == "snapkv_pp":
            if self.queries is None:
                raise RuntimeError("SnapKV requires query hook")
            groups = self.queries.shape[1] // h
            q = self.queries.reshape(b, h, groups, -1, d).float()
            k = self.keys[..., :candidates, :].float().unsqueeze(2)
            # Match repository's decode adaptation: mean over query window,
            # width-5 average pooling over physical slots, mean over GQA group.
            weights = (q @ k.transpose(-1, -2)) * d ** -0.5
            scores = weights.softmax(-1).mean(-2)
            scores = F.avg_pool1d(scores.reshape(b*h*groups, 1, candidates), 5, stride=1, padding=2)
            scores = scores.reshape(b, h, groups, candidates).mean(2)
        else:
            shape = (b, 1 if cfg.policy == "random_shared_pp" else h, candidates)
            if hasattr(self, 'generators'):
                scores = torch.stack([torch.rand(shape[1:], device=self.device, generator=g) for g in self.generators])
            else:
                scores = torch.rand(shape, device=self.device, generator=self.generator)
            scores = scores.expand(b, h, candidates).clone()
        if cfg.policy.endswith("_pp"):
            scores.masked_fill_(positions < self.prompt_length, float("inf"))
        chosen = scores.topk(keep_n, dim=-1).indices.sort(-1).values
        tail = torch.arange(candidates, n, device=self.device).view(1, 1, -1).expand(b, h, -1)
        keep = torch.cat((chosen, tail), dim=-1)
        kv_index = keep.unsqueeze(-1).expand(b, h, cfg.capacity, d)
        # Gather into fresh bounded temporaries before copying; avoid overlap.
        keys = self.keys.gather(-2, kv_index)
        values = self.values.gather(-2, kv_index)
        pos = self.positions.gather(-1, keep)
        self.key_storage[..., :cfg.capacity, :].copy_(keys)
        self.value_storage[..., :cfg.capacity, :].copy_(values)
        self.position_storage[..., :cfg.capacity].copy_(pos)
        self.length = cfg.capacity
        self.events += 1
        self._refresh_views()
        return True

    def batch_select_indices(self, indices):
        if not self.is_initialized:
            return
        if self.settings.policy == "none":
            return super().batch_select_indices(indices)
        self.key_storage = self.key_storage.index_select(0, indices)
        self.value_storage = self.value_storage.index_select(0, indices)
        self.position_storage = self.position_storage.index_select(0, indices)
        if hasattr(self, 'generators'):
            self.generators = [self.generators[i] for i in indices.tolist()]
        if self.queries is not None:
            self.queries = self.queries.index_select(0, indices)
        self._refresh_views()


class HybridEvictionCache(DynamicCache):
    def __init__(self, config, settings: EvictionConfig):
        super().__init__(config=config)
        text_config = config.get_text_config(decoder=True)
        self.full_indices = [i for i, kind in enumerate(text_config.layer_types) if kind == "full_attention"]
        for i in self.full_indices:
            self.layers[i] = EvictionLayer(settings, i)

    @property
    def eviction_events(self):
        return sum(self.layers[i].events for i in self.full_indices)


def install_eviction_hooks(model):
    """Instance-local hooks. Native DynamicCache calls are left untouched.

    Qwen3.5 computes q_norm before RoPE. SnapKV captures that tensor and
    applies the SAME upstream partial-RoPE helper to the captured query.
    No additional q_proj pass or forward replacement is needed.
    Returns handles so tests/callers can uninstall the hooks.
    """
    handles = []
    for module in model.modules():
        if not isinstance(module, Qwen3_5Attention):
            continue
        state = {}

        def pre_hook(mod, args, kwargs, state=state):
            cache = kwargs.get("past_key_values")
            state["capture"] = isinstance(cache, HybridEvictionCache) and cache.layers[mod.layer_idx].settings.policy == "snapkv_pp"

        def query_hook(mod, args, output, state=state):
            if state.get("capture"):
                state["query"] = output.detach()

        def post_hook(mod, args, kwargs, output, state=state):
            cache = kwargs.get("past_key_values")
            if not isinstance(cache, HybridEvictionCache):
                return
            layer = cache.layers[mod.layer_idx]
            if state.get("capture"):
                q = state.pop("query").transpose(1, 2)
                cos, sin = kwargs["position_embeddings"]
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                layer.record_queries(q)
            layer.evict_if_needed()

        handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(module.q_norm.register_forward_hook(query_hook))
        handles.append(module.register_forward_hook(post_hook, with_kwargs=True))
    if not handles:
        raise ValueError("No Qwen3.5 Full Attention modules found")
    return handles
