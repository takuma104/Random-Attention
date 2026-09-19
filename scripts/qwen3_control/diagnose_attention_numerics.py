"""Investigate the failed BF16 eager-reference gate without relaxing it.

Capture actual pre-o_proj attention, compare BF16 eager/SDPA with a same-input
FP32 oracle, then repeat official model forwards in FP32. No accuracy evaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen3_control.runtime import ROOT,load_model,prompt_tokens,source_hashes
import argparse
import hashlib
import json
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,apply_rotary_pos_emb,eager_attention_forward
from kvcompress.qwen3_control.cache import EvictionConfig,HybridEvictionCache,install_eviction_hooks
from scripts.qwen35.run_math import atomic_json


def error(a,b):
    a=a.float(); b=b.float(); d=a-b
    return dict(max_abs=d.abs().max().item(),mean_abs=d.abs().mean().item(),
                relative_l2=(d.norm()/b.norm().clamp_min(1e-12)).item())


@torch.inference_mode()
def run(dtype):
    model,tokenizer,info=load_model()
    if dtype==torch.float32: model.float()
    capacity=256; recent=64; steps=512
    prompt=prompt_tokens(tokenizer,'Compute 17 + 25.').expand(2,-1)
    final_length=prompt.shape[1]+steps-1
    inputs={}; snapshots=[]; handles=[]
    def diagnostic(mod,args,kwargs,output):
        cache=kwargs.get('past_key_values')
        if not isinstance(cache,HybridEvictionCache): return
        layer=cache.layers[mod.layer_idx]
        if layer.events==0 or (layer.logical_length%64 and layer.logical_length not in (321,final_length)): return
        actual_attention=inputs[mod.layer_idx]
        x=kwargs['hidden_states']; shape=x.shape[:-1]
        q=mod.q_norm(mod.q_proj(x).view(*shape,-1,mod.head_dim)).transpose(1,2)
        q,_=apply_rotary_pos_emb(q,q,*kwargs['position_embeddings'])
        mask=kwargs['attention_mask']
        assert mask is None, 'Expected unmasked single-query attention over retained past tokens'
        sdpa=F.scaled_dot_product_attention(q,layer.keys,layer.values,attn_mask=mask,
            dropout_p=0.,is_causal=False,scale=mod.scaling,enable_gqa=True).transpose(1,2).reshape(*shape,-1)
        torch.testing.assert_close(sdpa,actual_attention,rtol=0,atol=0)
        eager,_=eager_attention_forward(mod,q,layer.keys,layer.values,None,scaling=mod.scaling)
        eager=eager.reshape(*shape,-1)
        q32=q.float(); k32=layer.keys.float(); v32=layer.values.float(); w32=mod.o_proj.weight.float()
        oracle,_=eager_attention_forward(mod,q32,k32,v32,None,scaling=mod.scaling)
        oracle=oracle.reshape(*shape,-1)
        sdpa32=F.scaled_dot_product_attention(q32,k32,v32,dropout_p=0.,is_causal=False,
            scale=mod.scaling,enable_gqa=True).transpose(1,2).reshape(*shape,-1)
        torch.testing.assert_close(sdpa32,oracle,rtol=1e-4,atol=1e-5)
        projected_oracle=F.linear(oracle,w32)
        projected_eager=F.linear(eager,mod.o_proj.weight)
        projected_sdpa32=F.linear(sdpa32,w32)
        torch.testing.assert_close(projected_sdpa32,projected_oracle,rtol=1e-4,atol=1e-4)
        if dtype==torch.float32: torch.testing.assert_close(output[0],projected_oracle,rtol=1e-4,atol=1e-4)
        close=torch.isclose(projected_eager,output[0],rtol=.03,atol=.03)
        failures=(~close).nonzero().tolist()
        entry=dict(policy=layer.settings.policy,layer=mod.layer_idx,logical_length=layer.logical_length,
            n_elements=output[0].numel(),original_gate_failures=len(failures),
            sdpa_vs_eager=error(output[0],projected_eager),
            sdpa_vs_fp32_oracle=error(output[0],projected_oracle),
            eager_vs_fp32_oracle=error(projected_eager,projected_oracle),
            attention_sdpa_vs_fp32=error(actual_attention,oracle),
            attention_eager_vs_fp32=error(eager,oracle),
            fp32_sdpa_vs_eager=error(projected_sdpa32,projected_oracle),
            projection_only=error(output[0],F.linear(actual_attention.float(),w32)),
            examples=[dict(index=ix,sdpa=output[0][tuple(ix)].item(),eager=projected_eager[tuple(ix)].item(),
                           fp32_oracle=projected_oracle[tuple(ix)].item()) for ix in failures[:8]])
        snapshots.append(entry)
        if failures: print('ORIGINAL_GATE_FAILURE',str(dtype),entry['policy'],entry['layer'],entry['logical_length'],entry['examples'],flush=True)
    for m in model.modules():
        if isinstance(m,Qwen3Attention):
            def capture(mod,args,idx=m.layer_idx): inputs[idx]=args[0].detach()
            handles.append(m.o_proj.register_forward_pre_hook(capture))
            handles.append(m.register_forward_hook(diagnostic,with_kwargs=True))
    handles+=install_eviction_hooks(model)
    caches={'native':DynamicCache(config=model.config),
        'none':HybridEvictionCache(model.config,EvictionConfig('none',capacity,recent,(7,11)))}
    for p in ('random_pp','recency_pp','snapkv_pp'):
        caches[p]=HybridEvictionCache(model.config,EvictionConfig(p,capacity,recent,(7,11)))
    fill=tokenizer.encode(' 1 + 1 = 2. 2 + 2 = 4. 3 + 3 = 6.\n',add_special_tokens=False)
    forced=torch.tensor([fill*(steps//len(fill)+1)],device='cuda').expand(2,-1)
    current=prompt; pre_error=0.
    for step in range(steps):
        logits={name:model(current,past_key_values=cache,use_cache=True,logits_to_keep=1).logits for name,cache in caches.items()}
        torch.testing.assert_close(logits['native'],logits['none'],rtol=0,atol=0)
        if prompt.shape[1]+step<=capacity+recent:
            for p in ('random_pp','recency_pp','snapkv_pp'):
                pre_error=max(pre_error,(logits[p]-logits['native']).abs().max().item())
                torch.testing.assert_close(logits[p],logits['native'],rtol=0,atol=0)
        current=forced[:,step:step+1]
        if step%128==0: print('DIAGNOSTIC',str(dtype),'logical',prompt.shape[1]+step,flush=True)
    assert snapshots
    for h in handles: h.remove()
    metrics=['sdpa_vs_eager','sdpa_vs_fp32_oracle','eager_vs_fp32_oracle',
             'attention_sdpa_vs_fp32','attention_eager_vs_fp32','fp32_sdpa_vs_eager','projection_only']
    summary={metric:{field:max(s[metric][field] for s in snapshots) for field in ('max_abs','mean_abs','relative_l2')} for metric in metrics}
    return dict(dtype=str(dtype),model_revision=info['model_revision'],n_snapshots=len(snapshots),
        n_original_gate_failures=sum(s['original_gate_failures'] for s in snapshots),
        n_elements=sum(s['n_elements'] for s in snapshots),pre_eviction_logit_max_error=pre_error,
        all_layer_same_dtype_sdpa_replays_exact=True,summary=summary,snapshots=snapshots)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path); args=ap.parse_args()
    assert not torch.backends.cuda.matmul.allow_tf32
    results=[]
    for dtype in (torch.bfloat16,torch.float32):
        results.append(run(dtype)); torch.cuda.empty_cache()
        atomic_json(args.out,dict(status='partial' if dtype==torch.bfloat16 else 'complete',runs=results,
            source_hashes=source_hashes(),diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
    print(json.dumps([{k:r[k] for k in ('dtype','n_snapshots','n_original_gate_failures','n_elements','pre_eviction_logit_max_error','summary')} for r in results],indent=2))


if __name__=='__main__': main()
