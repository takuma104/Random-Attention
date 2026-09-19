"""Full-checkpoint B2 numeric gate for the isolated Qwen3 control."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen3_control.runtime import ROOT,load_model,prompt_tokens,generate_batch,source_hashes
import argparse
import hashlib
import json
import time
import torch
import torch.nn.functional as F
from scripts.qwen3_control.numerics import check_bf16_reference,LIMIT
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,apply_rotary_pos_emb,eager_attention_forward
from kvcompress.qwen3_control.cache import EvictionConfig,HybridEvictionCache,install_eviction_hooks
from kvcompress.qwen35.generate import state_bytes
from scripts.qwen35.run_math import atomic_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--steps',type=int)
    ap.add_argument('--capacity',type=int,default=256)
    ap.add_argument('--trace',choices=['calibration','heldout'],default='calibration'); args=ap.parse_args()
    assert not args.out.exists(), 'Use a new report path; preserve prior evidence'
    assert not torch.backends.cuda.matmul.allow_tf32
    model,tokenizer,info=load_model(); capacity=args.capacity; recent=64
    question='Compute 17 + 25.' if args.trace=='calibration' else 'Compute 23 * 19.'
    eviction_seeds=(7,11) if args.trace=='calibration' else (17,23)
    prompt=prompt_tokens(tokenizer,question).expand(2,-1)
    assert prompt.shape[1]<capacity-recent
    args.steps=args.steps or max(512,capacity+128-prompt.shape[1]+1)
    final_length=prompt.shape[1]+args.steps-1; assert final_length>capacity+recent
    naked=model(prompt,past_key_values=DynamicCache(config=model.config),use_cache=True,logits_to_keep=1).logits
    errors={p:[] for p in ('random_pp','recency_pp','snapkv_pp')}
    layers_checked={p:set() for p in errors}; inputs={}
    def reference_hook(mod,unused,kwargs,output):
        cache=kwargs.get('past_key_values')
        if not isinstance(cache,HybridEvictionCache): return
        layer=cache.layers[mod.layer_idx]; policy=layer.settings.policy
        if layer.events==0 or (layer.logical_length%64 and layer.logical_length not in (capacity+recent+1,final_length)): return
        x=kwargs['hidden_states']; shape=x.shape[:-1]
        q=mod.q_norm(mod.q_proj(x).view(*shape,-1,mod.head_dim)).transpose(1,2)
        q,_=apply_rotary_pos_emb(q,q,*kwargs['position_embeddings'])
        assert kwargs['attention_mask'] is None and mod.o_proj.bias is None
        actual_attention=inputs[mod.layer_idx]
        replay=F.scaled_dot_product_attention(q,layer.keys,layer.values,dropout_p=0.,
            is_causal=False,scale=mod.scaling,enable_gqa=True).transpose(1,2).reshape(*shape,-1)
        torch.testing.assert_close(replay,actual_attention,rtol=0,atol=0)
        oracle,_=eager_attention_forward(mod,q.float(),layer.keys.float(),layer.values.float(),None,scaling=mod.scaling)
        oracle=oracle.reshape(*shape,-1)
        by_head=lambda t:t.reshape(*shape,-1,mod.head_dim).transpose(1,2).flatten(0,1)
        pre=check_bf16_reference(by_head(actual_attention),by_head(oracle))
        post=check_bf16_reference(output[0],F.linear(oracle,mod.o_proj.weight.float()))
        old,_=eager_attention_forward(mod,q,layer.keys,layer.values,None,scaling=mod.scaling)
        old=F.linear(old.reshape(*shape,-1),mod.o_proj.weight)
        errors[policy].append(dict(pre_projection=pre,post_projection=post,
            original_gate_failures=(~torch.isclose(old,output[0],rtol=.03,atol=.03)).sum().item(),
            original_max_abs=(old.float()-output[0].float()).abs().max().item()))
        layers_checked[policy].add(mod.layer_idx)
    refs=[]
    for m in model.modules():
        if isinstance(m,Qwen3Attention):
            def capture(mod,args,idx=m.layer_idx): inputs[idx]=args[0].detach()
            refs.extend([m.o_proj.register_forward_pre_hook(capture),m.register_forward_hook(reference_hook,with_kwargs=True)])
    handles=install_eviction_hooks(model)
    caches={'native':DynamicCache(config=model.config),'none':HybridEvictionCache(model.config,EvictionConfig('none',capacity,recent,eviction_seeds))}
    for policy in errors: caches[policy]=HybridEvictionCache(model.config,EvictionConfig(policy,capacity,recent,eviction_seeds))
    text=' 1 + 1 = 2. 2 + 2 = 4. 3 + 3 = 6.\n' if args.trace=='calibration' else ' 7 - 3 = 4. 8 / 2 = 4. 5 * 3 = 15.\n'
    fill=tokenizer.encode(text,add_special_tokens=False)
    forced=torch.tensor([fill*(args.steps//len(fill)+1)],device='cuda').expand(2,-1)
    current=prompt; pre_checks=0; start=time.perf_counter()
    for step in range(args.steps):
        logits={name:model(current,past_key_values=cache,use_cache=True,logits_to_keep=1).logits for name,cache in caches.items()}
        if step==0: torch.testing.assert_close(naked,logits['native'],rtol=0,atol=0)
        torch.testing.assert_close(logits['native'],logits['none'],rtol=0,atol=0)
        logical=prompt.shape[1]+step
        for policy in errors:
            cache=caches[policy]; assert cache.get_seq_length()==logical
            assert cache.eviction_events==36*max(0,(logical-capacity)//recent)
            assert torch.isfinite(logits[policy]).all()
            if logical<=capacity+recent:
                torch.testing.assert_close(logits['native'],logits[policy],rtol=0,atol=0); pre_checks+=1
        current=forced[:,step:step+1]
        if step%128==0: print('NUMERIC logical=',logical,flush=True)
    cells={}
    for policy in errors:
        cache=caches[policy]; counts=state_bytes(cache)
        assert cache.full_indices==list(range(36)) and layers_checked[policy]==set(range(36))
        assert counts['recurrent']==counts['conv']==0
        assert counts['kv']==2*36*2*8*128*(capacity+recent)*2
        for layer in cache.layers:
            assert torch.all(layer.positions[...,1:]>layer.positions[...,:-1])
            assert torch.equal(layer.positions[...,:prompt.shape[1]],torch.arange(prompt.shape[1],device='cuda').expand(2,8,-1))
            assert torch.equal(layer.positions[...,-recent:],torch.arange(final_length-recent,final_length,device='cuda').expand(2,8,-1))
        summary={stage:{key:max(e[stage][key] for e in errors[policy]) for key in errors[policy][0][stage]}
                 for stage in ('pre_projection','post_projection')}
        cells[policy]=dict(reference_checks=len(errors[policy]),fp32_oracle_error=summary,
            original_gate_failures=sum(e['original_gate_failures'] for e in errors[policy]),
            original_max_abs=max(e['original_max_abs'] for e in errors[policy]),
            eviction_layer_events=cache.eviction_events,cache_bytes=counts)
    for h in refs: h.remove()
    del caches,cache,layer,logits,naked; inputs.clear()
    torch.cuda.empty_cache()
    replay={}; samples={}
    for policy,cap in [('native',0),('random_pp',1024)]:
        kw=dict(policy=policy,capacity=cap,recent=64,max_new_tokens=256,generation_seeds=[101,202],eviction_seeds=[303,404],heartbeat=0)
        first=generate_batch(model,prompt,**kw); second=generate_batch(model,prompt,**kw)
        assert [v[0] for v in first]==[v[0] for v in second]
        for tokens,r in first:
            assert not any(t in r['eos_ids'] for t in tokens[:-1])
            assert (tokens[-1] in r['eos_ids'])==(r['termination']=='eos')
            assert r['logical_cached_tokens']==prompt.shape[1]+len(tokens)-1
            assert r['cache_bytes']['recurrent']==r['cache_bytes']['conv']==0
        replay[policy]=[dict(tokens=len(t),termination=r['termination'],token_sha256=hashlib.sha256(json.dumps(t).encode()).hexdigest()) for t,r in first]
        samples[policy]=first
    assert [v[0] for v in samples['native']]==[v[0] for v in samples['random_pp']]
    sample_path=ROOT/'work/qwen3_control'/f'{args.out.stem}-samples.json'
    assert not sample_path.exists()
    atomic_json(sample_path,samples)
    questions=[json.loads(line) for line in Path(info['dataset_path']).read_text().splitlines()]
    lengths={q['experiment_id']:prompt_tokens(tokenizer,q['problem']).shape[1] for q in questions}
    for h in handles: h.remove()
    hashes=source_hashes()
    for source in (Path(__file__).resolve(),ROOT/'scripts/qwen3_control/numerics.py'):
        hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    report=dict(status='passed',model_revision=info['model_revision'],source_hashes=hashes,batch=2,
        capacity=capacity,recent=recent,prompt_tokens=prompt.shape[1],decode_steps=args.steps-1,
        trace=args.trace,eviction_seeds=eviction_seeds,logical_length=final_length,
        samples_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        native_hooks_inert=True,noop_logit_max_error=0.,pre_eviction_logit_max_error=0.,pre_eviction_checks=pre_checks,
        reference='same-input FP32 eager before and after o_proj',gate_version=2,
        relative_l2_limit=LIMIT,relative_linf_limit=LIMIT,same_input_sdpa_replay_exact=True,
        reference_units=dict(pre_projection='each batch row and query head',post_projection='each batch row'),
        cells=cells,sampling_replay=replay,no_eviction_sampling_matches_native=True,
        prompt_audit=dict(n=len(lengths),maximum=max(lengths.values()),ineligible_C1024=[p for p,n in lengths.items() if n>=960]),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        elapsed_seconds=time.perf_counter()-start)
    atomic_json(args.out,report); print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
