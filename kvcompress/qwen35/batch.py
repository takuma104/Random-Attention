"""Same-prompt rollout batching with independent per-row generation/eviction RNG.

Finished rows stay resident to keep batch shape stable; their extra cache updates
are ignored and cannot affect other rows. No cache compaction across batch rows.
Record each answer's state at its own EOS/cap, not its partner's finish time.
Per-answer elapsed_seconds is allocated GPU batch time (batch_wall/B); actual
request latency and shared peak memory are recorded separately.
"""
import time
import torch
from transformers.cache_utils import DynamicCache
from kvcompress.qwen35.cache import EvictionConfig, HybridEvictionCache
from kvcompress.qwen35.generate import SamplingConfig, candidate_distribution, state_bytes
from dataclasses import asdict


@torch.inference_mode()
def generate_batch(model, input_ids, *, policy, capacity, recent, max_new_tokens,
                   generation_seeds, eviction_seeds, sampling=SamplingConfig(), heartbeat=512):
    b=input_ids.shape[0]
    if b not in (1,2) or len(generation_seeds)!=b or len(eviction_seeds)!=b or max_new_tokens<1:
        raise ValueError('Supported batch is 1 or 2 with one pair of seeds per row')
    if b>1 and not torch.equal(input_ids,input_ids[:1].expand_as(input_ids)):
        raise ValueError('Initial batching supports the SAME prompt only')
    cache=DynamicCache(config=model.config) if policy=='native' else HybridEvictionCache(
        model.config,EvictionConfig(policy,capacity,recent,tuple(eviction_seeds)))
    generators=[torch.Generator(device=input_ids.device).manual_seed(s) for s in generation_seeds]
    config=model.config.get_text_config(decoder=True)
    seen=torch.zeros((b,config.vocab_size),dtype=torch.bool,device=input_ids.device)
    output=torch.empty((b,max_new_tokens),dtype=torch.long,device=input_ids.device)
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else list(eos or [])
    if not eos:
        raise ValueError('EOS missing')
    done=[False]*b
    records=[None]*b
    current=input_ids
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    ttft=None
    for step in range(max_new_tokens):
        result=model(current,past_key_values=cache,use_cache=True,logits_to_keep=1)
        ids,probs=candidate_distribution(result.logits[:,-1,:],seen,sampling)
        token=torch.full((b,1),eos[0],dtype=torch.long,device=input_ids.device)
        for row in range(b):
            if not done[row]:
                local=torch.multinomial(probs[row],1,generator=generators[row])
                token[row]=ids[row].gather(-1,local)
        output[:,step:step+1]=token
        seen.scatter_(1,token,True)
        token_ids=token.flatten().tolist()  # one host sync per decode step
        elapsed=time.perf_counter()-start
        if step==0:
            ttft=elapsed
        for row in range(b):
            if done[row]:
                continue
            if token_ids[row] in eos or step+1==max_new_tokens:
                reason='eos' if token_ids[row] in eos else 'length'
                byte_counts=state_bytes(cache)
                assert all(n%b==0 for n in byte_counts.values())
                records[row]=dict(generated_tokens=step+1,termination=reason,request_latency_seconds=elapsed,
                    ttft_seconds=ttft,cache_bytes={k:v//b for k,v in byte_counts.items()},
                    batch_cache_bytes_at_finish=byte_counts,
                    eviction_layer_events=getattr(cache,'eviction_events',0),
                    logical_cached_tokens=cache.get_seq_length(),
                    final_physical_lengths=[layer.length for layer in cache.layers if hasattr(layer,'length')],
                    sampling=asdict(sampling),eos_ids=eos,batch_size=b)
                done[row]=True
        if all(done):
            break
        current=token
        if heartbeat and (step+1)%heartbeat==0:
            print(f'  progress steps={step+1} active={sum(not x for x in done)} elapsed={elapsed:.1f}s',flush=True)
    torch.cuda.synchronize()
    total=time.perf_counter()-start
    outputs=[]
    for row,record in enumerate(records):
        n=record['generated_tokens']
        record.update(elapsed_seconds=total/b,batch_wall_seconds=total,
            output_tok_s=n/(total/b),
            decode_tok_s=(n-1)/(record['request_latency_seconds']-ttft) if n>1 else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        outputs.append((output[row,:n].cpu().tolist(),record))
    return outputs
