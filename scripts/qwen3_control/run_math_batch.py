"""Isolated, fixed-protocol Qwen3 pilot; reuse only frozen generic primitives."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen3_control.runtime import ROOT,load_model,prompt_tokens,generate_batch,source_hashes,artifacts,SAMPLING,stable_seed
from kvcompress.qwen3_control.cache import install_eviction_hooks
from scripts.qwen35.run_math import atomic_json,summarize
from scripts.qwen35.run_math_batch import grade_pending_group
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import platform
import subprocess
import time
import torch
import transformers
from transformers.models.qwen3 import modeling_qwen3,configuration_qwen3
from transformers.integrations import sdpa_attention
from transformers import cache_utils

CELLS=[('native',0,'native'),('random_pp',1024,'random_pp_C1024'),('random_pp',2048,'random_pp_C2048')]
PROTOCOL=ROOT/'docs/plans/qwen3-control-pilot-protocol.md'
GATES=[ROOT/'docs/plans'/f'qwen3-control-{name}.json' for name in
       ('validation-v2','validation-C1024-v2','validation-C2048-v2','generation-validation')]


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def verify_assets(info):
    assert digest(info['dataset_path'])==info['dataset_sha256']
    for name,record in info['files'].items():
        path=Path(info['model_path'])/name
        assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],name
    for path in GATES:
        gate=json.loads(path.read_text()); assert gate['status']=='passed'
        assert gate['model_revision']==info['model_revision']
        for name,sha in gate['source_hashes'].items(): assert digest(ROOT/name)==sha,name


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--phase',choices=['smoke','pilot'],required=True)
    args=parser.parse_args()
    args.limit,args.max_new_tokens=(2,8192) if args.phase=='smoke' else (50,32768)
    args.runs=2; args.cells='native,random_pp:1024,random_pp:2048'
    out=(ROOT/args.out).resolve()
    assert out.is_relative_to((ROOT/'results/qwen3_control').resolve()), 'Separate control output required'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=artifacts(); verify_assets(info)
    dataset=[json.loads(line) for line in Path(info['dataset_path']).read_text().splitlines()]
    mapping={r['experiment_id']:r for r in dataset}
    rows=[mapping[p] for p in info['pilot_ids'][:args.limit]]
    assert len(rows)==args.limit and len({r['experiment_id'] for r in rows})==args.limit
    hashes=source_hashes()
    for p in (Path(__file__).resolve(),ROOT/'scripts/qwen35/run_math_batch.py'):
        hashes[str(p.relative_to(ROOT))]=digest(p)
    installed={str(Path(m.__file__).resolve()):digest(m.__file__) for m in (modeling_qwen3,configuration_qwen3,sdpa_attention,cache_utils)}
    environment=subprocess.check_output(['uv','pip','freeze','--python',str(ROOT/'.venv/bin/python')],text=True)
    settings=dict(args=vars(args),batch_size=2,artifacts=info,sampling=asdict(SAMPLING),source_hashes=hashes,
        installed_source_hashes=installed,protocol_sha256=digest(PROTOCOL),
        validation_hashes={str(p.relative_to(ROOT)):digest(p) for p in GATES},environment=environment,
        torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),
        problem_ids=[r['experiment_id'] for r in rows],eos_ids=[151645,151643],full_attention_layers=36,
        batch_policy='Finished rows stay resident; no active-row compaction.',
        timing_policy='elapsed_seconds=batch_wall_seconds/2 is accounting, not request latency.')
    manifest=out/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())['settings']==settings, 'Resume source/settings mismatch'
    else:
        atomic_json(manifest,dict(settings=settings,created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            gpu=subprocess.check_output(['nvidia-smi'],text=True)))
    model,tokenizer,_=load_model(); handles=install_eviction_hooks(model)
    prompts={r['experiment_id']:prompt_tokens(tokenizer,r['problem']).expand(2,-1) for r in rows}
    assert all(p.shape[1]<960 and p.shape[1]+args.max_new_tokens<model.config.max_position_embeddings for p in prompts.values())
    warm=prompt_tokens(tokenizer,'What is 1+1?').expand(2,-1)
    generate_batch(model,warm,policy='native',capacity=0,recent=64,max_new_tokens=4,
                   generation_seeds=[1,2],eviction_seeds=[3,4],heartbeat=0)
    del warm
    for index,row in enumerate(rows):
        pid=row['experiment_id']; key=hashlib.sha256(pid.encode()).hexdigest()[:20]; ids=prompts[pid]
        for policy,capacity,name in CELLS:
            paths=[out/name/f'{key}_r{r}.json' for r in (0,1)]
            pending=out/name/f'{key}_g0.batch-ungraded'
            if all(p.exists() for p in paths):
                for run,path in enumerate(paths):
                    existing=json.loads(path.read_text())
                    assert existing['problem_id']==pid and existing['run']==run
                    assert existing['status']=='complete' and not existing['grading_errors']
                if pending.exists(): grade_pending_group(pending,paths)
                continue
            if not pending.exists() and any(p.exists() for p in paths):
                raise RuntimeError(f'Partially graded group lacks journal: {pending}')
            if not pending.exists():
                print(f'START {name} {pid} runs=[0,1] prompt={ids.shape[1]} batch=2',flush=True)
                gseeds=[stable_seed(pid,r,'generation') for r in (0,1)]
                eseeds=[stable_seed(pid,r,'eviction') for r in (0,1)]
                torch.cuda.empty_cache(); started=time.time()
                outputs=generate_batch(model,ids,policy=policy,capacity=capacity,recent=64,
                    max_new_tokens=args.max_new_tokens,generation_seeds=gseeds,eviction_seeds=eseeds,sampling=SAMPLING)
                raw=[]
                for j,(tokens,metrics) in enumerate(outputs):
                    raw.append(dict(metrics,problem_id=pid,source_index=row['source_index'],run=j,
                        policy=policy,capacity=capacity,recent=64,prompt_tokens=ids.shape[1],prompt_token_ids=ids[j].tolist(),
                        gold=row['answer'],level=row.get('level'),subject=row.get('subject'),model_revision=info['model_revision'],
                        generation_seed=gseeds[j],eviction_seed=eseeds[j],status='generated',started_at=started,
                        token_ids=tokens,completion=tokenizer.decode(tokens,skip_special_tokens=False),
                        batch_id=f'{name}/{key}_g0',batch_row=j))
                atomic_json(pending,raw)
            else:
                saved=json.loads(pending.read_text())
                assert [r['problem_id'] for r in saved]==[pid,pid] and [r['run'] for r in saved]==[0,1]
            grade_pending_group(pending,paths)
            summarize(out,[c[2] for c in CELLS],len(rows)*2)
        print(f'PROBLEM_COMPLETE {index+1}/{len(rows)}',flush=True)
    for h in handles: h.remove()
    print('FINISHED',json.dumps(summarize(out,[c[2] for c in CELLS],len(rows)*2)),flush=True)
    lock.close()


if __name__=='__main__': main()
