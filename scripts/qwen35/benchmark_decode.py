"""Fixed teacher-forced decode workload with allocator controls, not serving QPS.

Warm full histories are produced by the real model (no synthetic cache injection).
All methods see identical token IDs and logical measurement windows at each batch.
Throughput mode synchronizes window boundaries; synchronized mode also barriers
every step. Both include model+LM head but exclude sampling, grading and EOS.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen35.runtime import ROOT,load_model,kernel_provenance
import argparse
import fcntl
import hashlib
import json
import subprocess
import time
import torch
from transformers.cache_utils import DynamicCache
from kvcompress.qwen35.cache import EvictionConfig,HybridEvictionCache,install_eviction_hooks
from kvcompress.qwen35.generate import state_bytes
from scripts.qwen35.run_math import atomic_json,source_hashes
from scripts.qwen35.run_retrieval_probe import INSTRUCTION,make_case

DEFAULT_CELLS='native_dynamic,native_preallocated,random_pp:1024,random_pp:2048,recency_pp:1024,snapkv_pp:1024'


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi','-i','0',
        '--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
        '--format=csv,noheader,nounits'],text=True).strip()


def parse_cell(spec,maximum):
    if spec=='native_dynamic': return spec,None,None
    if spec=='native_preallocated': return spec,'random_pp',maximum
    policy,_,capacity=spec.partition(':')
    assert policy in ('random_pp','recency_pp','snapkv_pp') and int(capacity)>64
    return f'{policy}_C{capacity}',policy,int(capacity)


def schedule(contexts,steps,repeats,modes):
    assert steps>0 and repeats>0 and contexts==sorted(set(contexts))
    assert modes and len(modes)==len(set(modes)) and set(modes)<={'throughput','synchronized'}
    width=steps*repeats*len(modes)
    assert all(a+width<=b for a,b in zip(contexts,contexts[1:])), 'Measurement windows overlap'
    return contexts[-1]+width


@torch.inference_mode()
def run_cell(model,prompt,forced,*,spec,maximum,contexts,steps,repeats,modes,reference):
    b=prompt.shape[0]
    name,policy,capacity=parse_cell(spec,maximum)
    cache=DynamicCache(config=model.config) if policy is None else HybridEvictionCache(
        model.config,EvictionConfig(policy,capacity,64,tuple(7000+i for i in range(b))))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    wall_start=time.perf_counter(); before_gpu=gpu_snapshot()
    result=model(prompt,past_key_values=cache,use_cache=True,logits_to_keep=1)
    position=prompt.shape[1]; records=[]; reference_out={}; overall_peak=0
    for context in contexts:
        while position<context:
            offset=position-prompt.shape[1]
            result=model(forced[:,offset:offset+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
            position+=1
            if position%4096==0:
                torch.cuda.synchronize()
                print(f'  warm_history {name} B{b} logical={position}',flush=True)
        for repeat in range(repeats):
            for mode in modes:
                assert cache.get_seq_length()==position
                torch.cuda.synchronize()
                overall_peak=max(overall_peak,torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
                start_position=position
                start=time.perf_counter()
                for _ in range(steps):
                    offset=position-prompt.shape[1]
                    result=model(forced[:,offset:offset+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
                    position+=1
                    if mode=='synchronized': torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                peak=torch.cuda.max_memory_allocated(); overall_peak=max(overall_peak,peak)
                r=dict(method=name,batch=b,nominal_context=context,repeat=repeat,mode=mode,
                    logical_start=start_position,logical_end=position,steps=steps,
                    elapsed_seconds=elapsed,ms_per_step=1000*elapsed/steps,tokens_per_second=b*steps/elapsed,
                    peak_allocated_bytes=peak,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    cache_bytes=state_bytes(cache),eviction_layer_events=getattr(cache,'eviction_events',0))
                assert cache.get_seq_length()==position
                expected=0 if name.startswith('native_') else 8*max(0,(position-capacity)//64)
                assert r['eviction_layer_events']==expected
                # Untimed CPU copies: compare full-vocabulary logits at matched
                # logical positions, not just selected labels or sampled tokens.
                logits=result.logits[:,-1,:].detach().cpu()
                assert torch.isfinite(logits).all()
                key=f'{context}:{repeat}:{mode}'
                if name=='native_dynamic': reference_out[key]=logits
                if reference is not None and (name=='native_preallocated' or expected==0):
                    torch.testing.assert_close(logits,reference[key],rtol=0,atol=0)
                    r['no_eviction_full_logit_max_error']=0.
                elif name=='native_preallocated':
                    raise RuntimeError('Preallocated baseline needs preceding native_dynamic references')
                records.append(r)
                print(f'MEASURE {name} B{b} context={context} {mode} repeat={repeat} '
                      f'ms={r["ms_per_step"]:.3f} tok_s={r["tokens_per_second"]:.1f}',flush=True)
    torch.cuda.synchronize()
    report=dict(status='complete',method=name,batch=b,backing_capacity=capacity,
                reserved_slots_per_head=capacity+64 if capacity is not None else None,
                total_wall_seconds=time.perf_counter()-wall_start,
                overall_peak_allocated_bytes=overall_peak,gpu_before=before_gpu,gpu_after=gpu_snapshot(),records=records)
    del result,cache
    return report,reference_out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--contexts',default='8192,16384,32768')
    ap.add_argument('--batches',default='1,2,8')
    ap.add_argument('--steps',type=int,default=128)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--modes',default='throughput,synchronized')
    ap.add_argument('--cells',default=DEFAULT_CELLS)
    args=ap.parse_args()
    contexts=[int(x) for x in args.contexts.split(',')]
    batches=[int(x) for x in args.batches.split(',')]
    modes=args.modes.split(','); cells=args.cells.split(',')
    maximum=schedule(contexts,args.steps,args.repeats,modes)
    assert batches and min(batches)>0
    if 'native_preallocated' in cells:
        assert 'native_dynamic' in cells and cells.index('native_dynamic')<cells.index('native_preallocated')
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'.run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    hashes=source_hashes()
    for path in [Path(__file__).resolve(),ROOT/'scripts/qwen35/run_retrieval_probe.py']:
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    settings=dict(args={**vars(args),'out':str(args.out)},source_hashes=hashes,torch=torch.__version__,
                  kernels=kernel_provenance(),maximum_logical_length=maximum,
                  artifacts=json.loads((ROOT/'work/qwen35/artifacts.json').read_text()))
    manifest=args.out/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())['settings']==settings,'Resume settings/source mismatch'
    else:
        atomic_json(manifest,dict(settings=settings,created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gpu=gpu_snapshot()))
    model,tokenizer,_=load_model(); handles=install_eviction_hooks(model)
    base_prompt=tokenizer.apply_chat_template([dict(role='user',content=INSTRUCTION)],tokenize=True,
        add_generation_prompt=True,enable_thinking=True,return_tensors='pt',return_dict=True)['input_ids'].to('cuda')
    assert contexts[0]>base_prompt.shape[1]
    # A fixed natural-language arithmetic notebook, produced by the same data
    # helper as calibration; no injected/random KV or synthetic recurrent state.
    case=make_case(tokenizer,'calibration',0,maximum)
    history=(case['fact_ids']+case['filler_ids'])[:maximum-base_prompt.shape[1]]
    ids=torch.tensor([history],device='cuda')
    atomic_json(args.out/'input.json',dict(prompt_ids=base_prompt[0].tolist(),forced_ids=history,
        input_sha256=hashlib.sha256(json.dumps([base_prompt[0].tolist(),history]).encode()).hexdigest()))
    for batch in batches:
        reference=None
        for spec in cells:
            name,_,_=parse_cell(spec,maximum)
            path=args.out/f'B{batch}'/f'{name}.json'
            ref_path=args.out/f'B{batch}'/'native_logits.pt'
            if path.exists():
                saved=json.loads(path.read_text()); assert saved['status']=='complete'
                if name=='native_dynamic':
                    assert hashlib.sha256(ref_path.read_bytes()).hexdigest()==saved['reference_sha256']
                    reference=torch.load(ref_path,map_location='cpu',weights_only=True)
                continue
            print(f'START {name} batch={batch}',flush=True)
            report,new_reference=run_cell(model,base_prompt.expand(batch,-1),ids.expand(batch,-1),
                spec=spec,maximum=maximum,contexts=contexts,steps=args.steps,repeats=args.repeats,modes=modes,reference=reference)
            if name=='native_dynamic':
                ref_path.parent.mkdir(parents=True,exist_ok=True)
                temporary=ref_path.with_suffix('.pt.tmp'); torch.save(new_reference,temporary); temporary.replace(ref_path)
                report['reference_sha256']=hashlib.sha256(ref_path.read_bytes()).hexdigest()
                reference=new_reference
            atomic_json(path,report)
    for h in handles: h.remove()
    lock.close(); print('FINISHED benchmark',flush=True)


if __name__=='__main__': main()
