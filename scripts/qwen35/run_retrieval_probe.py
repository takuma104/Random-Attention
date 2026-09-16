"""Teacher-forced delayed lookup ONLY in the generated region (not protected prompt).

Separate from the frozen math runner. Two identical traces per batch use two
independent eviction seeds; no token sampling. Retention is captured BEFORE the
last query token, because eviction hooks execute AFTER attention.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen35.runtime import ROOT,load_model,kernel_provenance
import argparse
import fcntl
import hashlib
import itertools
import json
import random
import subprocess
import time
import torch
from transformers.cache_utils import DynamicCache
from kvcompress.qwen35.cache import EvictionConfig,HybridEvictionCache,install_eviction_hooks
from kvcompress.qwen35.generate import stable_seed
from scripts.qwen35.run_math import atomic_json,source_hashes

INSTRUCTION=('Use the working notes to remember an arbitrary key-to-value assignment. '
             'Other notes are unrelated distractions. At the final lookup, select the '
             'option containing the value originally assigned to the requested key. '
             'Answer with exactly one option letter A, B, C, or D inside a box.')
VALUES=['amber','cobalt','ivory','jade','ochre','teal','coral','silver']


def make_case(tokenizer,phase,index,max_gap):
    case_id=f'{phase}-{index:03d}'
    rng=random.Random(stable_seed(case_id,0,'probe-data-v1'))
    key=f'KEY_{rng.randrange(100000,1000000)}'
    options=rng.sample(VALUES,4)
    target=index%4
    value=options[target]
    fact=f'First I record the assignment: key {key} has value {value}.\n'
    value_start=fact.index(value)
    encoded=tokenizer(fact,add_special_tokens=False,return_offsets_mapping=True)
    value_tokens=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if b>value_start and a<value_start+len(value)]
    assert value_tokens and value_tokens==list(range(value_tokens[0],value_tokens[-1]+1))
    sentences=[]
    for i in range(max_gap//10+100):
        a,b=rng.randrange(1,100),rng.randrange(1,100)
        sentences.append(f'Separate arithmetic check {i}: {a} plus {b} equals {a+b}. I move on to the next unrelated note.\n')
    filler_text=''.join(sentences)
    assert key not in filler_text and all(v not in filler_text.lower() for v in VALUES)
    filler=tokenizer.encode(filler_text,add_special_tokens=False)
    assert len(filler)>=max_gap
    query=(f'\nNow I look up the original value assigned to key {key}.\n'+
           '\n'.join(f'{label}. {v}' for label,v in zip('ABCD',options))+
           '\nThe matching option letter is my final answer.\n</think>\n\\boxed{')
    return dict(case_id=case_id,key=key,options=options,target=target,fact=fact,query=query,
        fact_ids=encoded['input_ids'],filler_ids=filler[:max_gap],
        query_ids=tokenizer.encode(query,add_special_tokens=False),
        value_offsets=[value_tokens[0],value_tokens[-1]+1])


def retained(cache,full_indices,prompt_length,fact_span,value_span,kv_heads):
    b=2
    fact_counts=[[] for _ in range(b)]
    value_counts=[[] for _ in range(b)]
    overlaps=[[] for _ in range(b)]
    for index in full_indices:
        layer=cache.layers[index]
        if not hasattr(layer,'positions'):
            for row in range(b):
                fact_counts[row].append([fact_span[1]-fact_span[0]]*kv_heads)
                value_counts[row].append([value_span[1]-value_span[0]]*kv_heads)
                overlaps[row].append(1.)
            continue
        positions=layer.positions.cpu().tolist()
        cutoff=cache.get_seq_length()-64
        for row in range(b):
            fact_counts[row].append([sum(fact_span[0]<=p<fact_span[1] for p in head) for head in positions[row]])
            value_counts[row].append([sum(value_span[0]<=p<value_span[1] for p in head) for head in positions[row]])
            older=[{p for p in head if prompt_length<=p<cutoff} for head in positions[row]]
            jaccards=[len(a&c)/len(a|c) if a|c else 1. for a,c in itertools.combinations(older,2)]
            overlaps[row].append(sum(jaccards)/len(jaccards))
    return dict(fact_token_counts=fact_counts,value_token_counts=value_counts,
                older_generated_head_jaccard_by_layer=overlaps,
                snapshot_logical_length=cache.get_seq_length(),
                eviction_layer_events_before_last_query=getattr(cache,'eviction_events',0))


@torch.inference_mode()
def score_trace(model,prompt,trace,*,policy,capacity,seeds,fact_span,value_span):
    cfg=model.config.get_text_config(decoder=True)
    full_indices=[i for i,kind in enumerate(cfg.layer_types) if kind=='full_attention']
    cache=DynamicCache(config=model.config) if policy=='native' else HybridEvictionCache(
        model.config,EvictionConfig(policy,capacity,64,tuple(seeds)))
    ids=torch.tensor([trace],device='cuda').expand(2,-1)
    torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    model(prompt,past_key_values=cache,use_cache=True,logits_to_keep=1)
    for i in range(len(trace)-1):
        model(ids[:,i:i+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
        if (i+1)%4096==0:
            torch.cuda.synchronize()
            print(f'  forced_steps={i+1} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    assert cache.get_seq_length()==prompt.shape[1]+len(trace)-1
    retention=retained(cache,full_indices,prompt.shape[1],fact_span,value_span,cfg.num_key_value_heads)
    result=model(ids[:,-1:],past_key_values=cache,use_cache=True,logits_to_keep=1)
    logits=result.logits[:,-1,:].float()
    assert torch.isfinite(logits).all()
    torch.cuda.synchronize()
    metrics=dict(elapsed_seconds=time.perf_counter()-started,
        peak_shared_allocated_bytes=torch.cuda.max_memory_allocated(),
        final_logical_length=cache.get_seq_length(),eviction_layer_events=getattr(cache,'eviction_events',0))
    return logits,retention,metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--phase',choices=['calibration','confirm'],default='calibration')
    ap.add_argument('--cases',type=int,default=8)
    ap.add_argument('--gaps',default='0,8192,16384')
    ap.add_argument('--cells',default='native')
    args=ap.parse_args()
    gaps=[int(x) for x in args.gaps.split(',')]
    assert args.cases>0 and gaps and min(gaps)>=0
    cells=[]
    for spec in args.cells.split(','):
        policy,_,cap=spec.partition(':')
        cells.append((policy,int(cap or 0),policy if policy=='native' else f'{policy}_C{cap}'))
    hashes=source_hashes()
    own=Path(__file__).resolve()
    hashes[str(own.relative_to(ROOT))]=hashlib.sha256(own.read_bytes()).hexdigest()
    artifacts=json.loads((ROOT/'work/qwen35/artifacts.json').read_text())
    settings=dict(args={**vars(args),'out':str(args.out)},instruction=INSTRUCTION,source_hashes=hashes,
                  artifacts=artifacts,kernels=kernel_provenance(),torch=torch.__version__,batch_size=2)
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'.run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=args.out/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())['settings']==settings,'Frozen probe settings changed'
    else:
        atomic_json(manifest,dict(settings=settings,created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()))
    model,tokenizer,_=load_model()
    handles=install_eviction_hooks(model)
    prompt=tokenizer.apply_chat_template([dict(role='user',content=INSTRUCTION)],tokenize=True,
        add_generation_prompt=True,enable_thinking=True,return_tensors='pt',return_dict=True)['input_ids'].to('cuda').expand(2,-1)
    labels=[tokenizer.encode(x,add_special_tokens=False) for x in 'ABCD']
    assert all(len(x)==1 for x in labels)
    label_ids=torch.tensor([x[0] for x in labels],device='cuda')
    for index in range(args.cases):
        case=make_case(tokenizer,args.phase,index,max(gaps))
        p=prompt.shape[1]
        assert case['key'] not in INSTRUCTION
        fact_span=[p,p+len(case['fact_ids'])]
        value_span=[p+x for x in case['value_offsets']]
        seeds=[stable_seed(case['case_id'],run,'probe-eviction-v1') for run in (0,1)]
        for gap in gaps:
            trace=case['fact_ids']+case['filler_ids'][:gap]+case['query_ids']
            trace_hash=hashlib.sha256(json.dumps(trace).encode()).hexdigest()
            reference=None
            for policy,capacity,name in cells:
                path=args.out/name/f'{case["case_id"]}_gap{gap}.json'
                if path.exists():
                    saved=json.loads(path.read_text()); assert saved['trace_sha256']==trace_hash
                    if policy=='native': reference=torch.tensor(saved['label_logits'])
                    continue
                print(f'START {name} {case["case_id"]} gap={gap} protected_prompt={p} fact_span={fact_span}',flush=True)
                logits,retention,metrics=score_trace(model,prompt,trace,policy=policy,capacity=capacity,
                    seeds=seeds,fact_span=fact_span,value_span=value_span)
                scores=logits.index_select(-1,label_ids).cpu()
                probs=scores.softmax(-1)
                predictions=scores.argmax(-1).tolist()
                if policy=='native':
                    torch.testing.assert_close(scores[0],scores[1],rtol=0,atol=0)
                    reference=scores
                pre_eviction_error=None
                if policy!='native' and metrics['eviction_layer_events']==0 and reference is not None:
                    pre_eviction_error=(scores-reference).abs().max().item()
                    torch.testing.assert_close(scores,reference,rtol=0,atol=0)
                record=dict(metrics,**retention,case_id=case['case_id'],phase=args.phase,gap_tokens=gap,
                    policy=policy,capacity=capacity,batch_size=2,eviction_seeds=seeds,
                    prompt_length=p,prompt_ids=prompt[0].tolist(),fact_span=fact_span,value_span=value_span,
                    query_length=len(case['query_ids']),trace_sha256=trace_hash,trace_ids=trace,
                    fact=case['fact'],query=case['query'],options=case['options'],target=case['target'],
                    label_token_ids=label_ids.tolist(),label_logits=scores.tolist(),conditional_probabilities=probs.tolist(),
                    label_probability_mass=logits.softmax(-1).index_select(-1,label_ids).sum(-1).tolist(),
                    predictions=predictions,correct=[x==case['target'] for x in predictions],
                    pre_eviction_label_logit_max_error=pre_eviction_error)
                atomic_json(path,record)
                print(f'DONE {name} {case["case_id"]} gap={gap} correct={record["correct"]} '
                      f'p_target={probs[:,case["target"]].tolist()}',flush=True)
                del logits
    for h in handles: h.remove()
    lock.close()
    print('FINISHED probe',flush=True)


if __name__=='__main__':
    main()
