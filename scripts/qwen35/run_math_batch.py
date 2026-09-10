"""Same-prompt two-rollout runner; atomic batch recovery and per-answer grading.

B1 pilot is NOT pooled with B2 main because BF16 batch-shape rounding can change
sampled trajectories. Within this runner all methods share B=2 and row seeds.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen35.runtime import ROOT,load_model,prompt_tokens,kernel_provenance
import argparse
import fcntl
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
import torch
import transformers
from kvcompress.qwen35.cache import install_eviction_hooks
from kvcompress.qwen35.batch import generate_batch
from kvcompress.qwen35.generate import SamplingConfig,stable_seed
from scripts.qwen35.run_math import atomic_json,source_hashes,summarize

DEFAULT_CELLS='native,random_pp:1024,random_pp:2048,recency_pp:1024,snapkv_pp:1024'


def grade_pending_group(pending, paths):
    """May restart with some individual answers already graded. Never regenerate."""
    rows=json.loads(pending.read_text())
    if len(rows)!=len(paths):
        raise RuntimeError('Corrupt pending batch size')
    for raw,path in zip(rows,paths):
        if path.exists():
            existing=json.loads(path.read_text())
            for field in ('problem_id','run','token_ids','generation_seed','eviction_seed'):
                assert existing[field]==raw[field], f'Pending/graded mismatch: {path}'
            if existing.get('grading_errors'):
                raise RuntimeError(f'Unresolved grader error {path}')
            continue
        graded=subprocess.run([sys.executable,'-m','kvcompress.qwen35.grade'],
            input=json.dumps(dict(completion=raw['completion'],gold=raw['gold'],termination=raw['termination'])),
            text=True,capture_output=True,timeout=30,cwd=ROOT,check=True)
        grade=json.loads(graded.stdout)
        final=dict(raw,**grade,status='complete')
        atomic_json(path,final)
        if grade['grading_errors']:
            raise RuntimeError(f'Grading error persisted in {path}: {grade["grading_errors"]}')
        print(f"DONE {path.parent.name} {raw['problem_id']} run={raw['run']} correct={grade['final_correct']} "
              f"tokens={raw['generated_tokens']} end={raw['termination']} latency={raw['request_latency_seconds']:.1f}s",flush=True)
    pending.unlink()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--subset',choices=['pilot','all'],default='pilot')
    parser.add_argument('--limit',type=int,default=2)
    parser.add_argument('--runs',type=int,default=2)
    parser.add_argument('--max-new-tokens',type=int,default=8192)
    parser.add_argument('--cells',default=DEFAULT_CELLS)
    args=parser.parse_args()
    if args.runs<2 or args.runs%2 or args.limit<1:
        raise ValueError('B2 requires positive limit and an even number of runs')
    out=ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    artifacts=json.loads((ROOT/'work/qwen35/artifacts.json').read_text())
    rows=[json.loads(s) for s in (ROOT/'work/qwen35/math500.jsonl').read_text().splitlines()]
    if args.subset=='pilot':
        mapping={r['experiment_id']:r for r in rows}
        rows=[mapping[p] for p in artifacts['pilot_ids']]
    rows=rows[:args.limit]
    cells=[]
    for spec in args.cells.split(','):
        policy,_,cap=spec.partition(':')
        capacity=int(cap) if cap else 0
        if policy!='native' and capacity<=64:
            raise ValueError('Capacity>64 required')
        if policy=='native' and capacity:
            raise ValueError('Native has no capacity')
        name='native' if policy=='native' else f'{policy}_C{capacity}'
        cells.append((policy,capacity,name))
    sampling=SamplingConfig()
    hashes=source_hashes()
    own_file=Path(__file__).resolve()
    hashes[str(own_file.relative_to(ROOT))]=hashlib.sha256(own_file.read_bytes()).hexdigest()
    settings=dict(args=vars(args),batch_size=2,artifacts=artifacts,sampling=asdict(sampling),
                  source_hashes=hashes,torch=torch.__version__,transformers=transformers.__version__,
                  python=platform.python_version(),kernels=kernel_provenance(),problem_ids=[r['experiment_id'] for r in rows],
                  batch_policy='Finished rows stay resident; no active-row compaction.',
                  timing_policy='elapsed_seconds=batch_wall_seconds/2 is allocated GPU time, not request latency.')
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['settings']!=settings:
            raise RuntimeError('Resume source/settings mismatch: use a new directory')
    else:
        atomic_json(manifest,dict(settings=settings,created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            environment=subprocess.check_output(['uv','pip','freeze','--python',str(ROOT/'.venv/bin/python')],text=True),
            gpu=subprocess.check_output(['nvidia-smi'],text=True)))
    model,tokenizer,_=load_model()
    handles=install_eviction_hooks(model)
    warm=prompt_tokens(tokenizer,'What is 1+1?').expand(2,-1)
    generate_batch(model,warm,policy='native',capacity=0,recent=64,max_new_tokens=4,
                   generation_seeds=[1,2],eviction_seeds=[3,4],heartbeat=0)
    del warm
    for row in rows:
        pid=row['experiment_id']
        key=hashlib.sha256(pid.encode()).hexdigest()[:20]
        ids=prompt_tokens(tokenizer,row['problem']).expand(2,-1)
        for first_run in range(0,args.runs,2):
            run_ids=[first_run,first_run+1]
            for policy,capacity,name in cells:
                paths=[out/name/f'{key}_r{r}.json' for r in run_ids]
                pending=out/name/f'{key}_g{first_run}.batch-ungraded'
                if all(p.exists() for p in paths):
                    for run,path in zip(run_ids,paths):
                        existing=json.loads(path.read_text())
                        assert existing['problem_id']==pid and existing['run']==run
                        assert existing['status']=='complete' and not existing['grading_errors']
                    if pending.exists():
                        grade_pending_group(pending,paths)
                    continue
                if not pending.exists() and any(p.exists() for p in paths):
                    raise RuntimeError(f'Partly graded group lacks recovery journal: {pending}')
                if not pending.exists():
                    if policy!='native' and ids.shape[1]>=capacity-64:
                        raise ValueError(f'Prompt budget overflow: {pid} C{capacity}')
                    print(f'START {name} {pid} runs={run_ids} prompt={ids.shape[1]} batch=2',flush=True)
                    gseeds=[stable_seed(pid,r,'generation') for r in run_ids]
                    eseeds=[stable_seed(pid,r,'eviction') for r in run_ids]
                    torch.cuda.empty_cache()
                    started=time.time()
                    outputs=generate_batch(model,ids,policy=policy,capacity=capacity,recent=64,
                        max_new_tokens=args.max_new_tokens,generation_seeds=gseeds,eviction_seeds=eseeds,sampling=sampling)
                    raw=[]
                    for j,(tokens,metrics) in enumerate(outputs):
                        raw.append(dict(metrics,problem_id=pid,source_index=row['source_index'],run=run_ids[j],
                            policy=policy,capacity=capacity,recent=64,prompt_tokens=ids.shape[1],
                            prompt_token_ids=ids[j].tolist(),gold=row['answer'],level=row.get('level'),subject=row.get('subject'),
                            generation_seed=gseeds[j],eviction_seed=eseeds[j],status='generated',started_at=started,
                            token_ids=tokens,completion=tokenizer.decode(tokens,skip_special_tokens=False),
                            batch_id=f'{name}/{key}_g{first_run}',batch_row=j))
                    # Both expensive rollouts are durable BEFORE either is graded.
                    atomic_json(pending,raw)
                else:
                    saved=json.loads(pending.read_text())
                    assert [r['problem_id'] for r in saved]==[pid,pid]
                    assert [r['run'] for r in saved]==run_ids
                grade_pending_group(pending,paths)
                summarize(out,[c[2] for c in cells],len(rows)*args.runs)
    for h in handles:
        h.remove()
    print('FINISHED',json.dumps(summarize(out,[c[2] for c in cells],len(rows)*args.runs)),flush=True)
    lock.close()


if __name__=='__main__':
    main()
