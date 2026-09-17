"""Separate B2 large-budget numeric gate; never changes the generation harness."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen35.runtime import ROOT,load_model,kernel_provenance
import argparse
import hashlib
import json
import time
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention,apply_rotary_pos_emb,eager_attention_forward
from kvcompress.qwen35.cache import EvictionConfig,HybridEvictionCache,install_eviction_hooks
from scripts.qwen35.run_math import source_hashes,atomic_json
from scripts.qwen35.run_retrieval_probe import INSTRUCTION,make_case


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path); args=ap.parse_args()
    model,tokenizer,artifacts=load_model()
    errors=[]
    def reference_hook(mod,unused,kwargs,output):
        cache=kwargs.get('past_key_values')
        if not isinstance(cache,HybridEvictionCache): return
        layer=cache.layers[mod.layer_idx]
        if layer.events==0 or kwargs['hidden_states'].shape[1]!=1: return
        x=kwargs['hidden_states']; shape=x.shape[:-1]
        q,gate=torch.chunk(mod.q_proj(x).view(*shape,-1,mod.head_dim*2),2,dim=-1)
        q=mod.q_norm(q).transpose(1,2)
        q,_=apply_rotary_pos_emb(q,q,*kwargs['position_embeddings'])
        a,_=eager_attention_forward(mod,q,layer.keys,layer.values,None,scaling=mod.scaling)
        a=mod.o_proj(a.reshape(*shape,-1)*gate.reshape(*shape,-1).sigmoid())
        errors.append((a.float()-output[0].float()).abs().max().item())
        torch.testing.assert_close(a,output[0],rtol=.03,atol=.03)
    # Reference must run BEFORE the post-attention compaction hook.
    refs=[m.register_forward_hook(reference_hook,with_kwargs=True) for m in model.modules() if isinstance(m,Qwen3_5Attention)]
    handles=install_eviction_hooks(model)
    prompt=tokenizer.apply_chat_template([dict(role='user',content=INSTRUCTION)],tokenize=True,
        add_generation_prompt=True,enable_thinking=True,return_tensors='pt',return_dict=True)['input_ids'].to('cuda').expand(2,-1)
    case=make_case(tokenizer,'calibration',0,8320)
    forced=torch.tensor([case['fact_ids']+case['filler_ids']],device='cuda').expand(2,-1)
    reports=[]; started=time.perf_counter()
    for capacity in (4096,8192):
        errors.clear(); checkpoints=[]
        native=DynamicCache(config=model.config)
        cache=HybridEvictionCache(model.config,EvictionConfig('random_pp',capacity,64,(7000,7001)))
        current=prompt; logical=prompt.shape[1]
        while True:
            a=model(current,past_key_values=native,use_cache=True,logits_to_keep=1).logits
            b=model(current,past_key_values=cache,use_cache=True,logits_to_keep=1).logits
            assert native.get_seq_length()==cache.get_seq_length()==logical
            assert torch.isfinite(b).all()
            if logical<=capacity+64 and (logical%1024==0 or logical in (prompt.shape[1],capacity+64)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                checkpoints.append(logical)
            if logical%1024==0: print(f'VALIDATE C{capacity} logical={logical} reference_checks={len(errors)}',flush=True)
            if logical==capacity+128: break
            offset=logical-prompt.shape[1]
            current=forced[:,offset:offset+1]; logical+=1
        assert cache.eviction_events==16 and len(errors)==512
        reports.append(dict(capacity=capacity,batch=2,final_logical_length=logical,eviction_layer_events=cache.eviction_events,
            exact_pre_eviction_logit_checkpoints=checkpoints,max_pre_eviction_logit_error=0.,
            reference_attention_checks=len(errors),max_reference_attention_error=max(errors),rtol=.03,atol=.03))
        del a,b,native,cache
        torch.cuda.empty_cache()
    for h in refs+handles: h.remove()
    hashes=source_hashes()
    for path in (Path(__file__).resolve(),ROOT/'scripts/qwen35/run_retrieval_probe.py'):
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(status='passed',model_revision=artifacts['model_revision'],source_hashes=hashes,
        kernels=kernel_provenance(),elapsed_seconds=time.perf_counter()-started,checks=reports)
    atomic_json(args.out,report); print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
