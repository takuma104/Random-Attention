"""Resumable, manifest-checked Qwen3.5 MATH500 experiment (single GPU job).

Each answer is atomically saved in its own JSON; no shard-position grading.
Do not change code mid-run. Source hashes are checked on resume.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen35.runtime import ROOT, load_model, prompt_tokens, kernel_provenance
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
import subprocess
import time
import torch
import transformers
from kvcompress.qwen35.cache import install_eviction_hooks
from kvcompress.qwen35.generate import SamplingConfig, generate_one, stable_seed

DEFAULT_CELLS = "native,random_pp:1024,recency_pp:1024,random_pp:2048,recency_pp:2048,random_pp:4096,recency_pp:4096"


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open('w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def source_hashes():
    paths = sorted((ROOT / "kvcompress/qwen35").glob("*.py")) + [Path(__file__)]
    paths += [ROOT / "kvcompress/harness/Utils" / name for name in ("grader.py", "parser.py", "math_normalization.py")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def summarize(out, cells, expected):
    report = {}
    for name in cells:
        records = [json.loads(p.read_text()) for p in sorted((out/name).glob('*.json'))]
        ok = [r for r in records if r['status'] == 'complete']
        avg = lambda field: sum(r[field] for r in ok)/len(ok) if ok else None
        report[name] = dict(n_complete=len(ok), n_expected=expected,
                            budget_ineligible=sum(r['status']=='budget_ineligible' for r in records),
                            n_grading_errors=sum(bool(r['grading_errors']) for r in ok),
                            final_accuracy=avg('final_correct'), paper_accuracy=avg('paper_correct'),
                            final_accuracy_terminated=avg('final_correct_terminated'),
                            mean_generated_tokens=avg('generated_tokens'),
                            cap_rate=sum(r['termination']=='length' for r in ok)/len(ok) if ok else None,
                            eviction_active_rate=sum(r['eviction_layer_events']>0 for r in ok)/len(ok) if ok else None,
                            total_seconds=sum(r['elapsed_seconds'] for r in ok),
                            aggregate_tok_s=sum(r['generated_tokens'] for r in ok)/sum(r['elapsed_seconds'] for r in ok) if ok else None,
                            peak_allocated_bytes=max((r['peak_allocated_bytes'] for r in ok),default=None))
    atomic_json(out/'summary.json', report)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--subset',choices=['pilot','all'],default='pilot')
    ap.add_argument('--limit',type=int,default=50)
    ap.add_argument('--runs',type=int,default=1)
    ap.add_argument('--max-new-tokens',type=int,default=8192)
    ap.add_argument('--cells',default=DEFAULT_CELLS)
    args=ap.parse_args()
    if args.limit<1 or args.runs<1:
        raise ValueError('Positive limit/runs required')
    out=ROOT/args.out
    artifacts=json.loads((ROOT/'work/qwen35/artifacts.json').read_text())
    rows=[json.loads(line) for line in (ROOT/'work/qwen35/math500.jsonl').read_text().splitlines()]
    if args.subset=='pilot':
        by_id={r['experiment_id']:r for r in rows}
        rows=[by_id[i] for i in artifacts['pilot_ids']]
    rows=rows[:args.limit]
    cells=[]
    for cell in args.cells.split(','):
        policy, _, cap=cell.partition(':')
        capacity=int(cap) if cap else 0
        if policy != 'native' and capacity<=64:
            raise ValueError('Explicit capacity >64 required')
        if policy=='native' and capacity:
            raise ValueError('Native has no capacity')
        cells.append((policy,capacity,'native' if policy=='native' else f'{policy}_C{capacity}'))
    sampling=SamplingConfig()
    frozen=dict(args=vars(args), artifacts=artifacts, sampling=asdict(sampling), batch_size=1,
                source_hashes=source_hashes(), torch=torch.__version__, transformers=transformers.__version__,
                python=platform.python_version(), kernels=kernel_provenance(),
                problem_ids=[r['experiment_id'] for r in rows])
    manifest=out/'manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous['settings']!=frozen:
            raise RuntimeError('Resume settings/source mismatch. Use a NEW output directory.')
    else:
        atomic_json(manifest,dict(settings=frozen, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                  git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                                  environment=subprocess.check_output(['uv','pip','freeze','--python',str(ROOT/'.venv/bin/python')],text=True),
                                  gpu=subprocess.check_output(['nvidia-smi'],text=True)))
    model,tokenizer,_=load_model()
    hooks=install_eviction_hooks(model)
    # Warm up kernels independently of evaluation records.
    warm=prompt_tokens(tokenizer,'What is 1+1?')
    generate_one(model,warm,policy='native',capacity=0,recent=64,max_new_tokens=4,
                 generation_seed=1,eviction_seed=2,heartbeat=0)
    del warm
    for row in rows:
        ids=prompt_tokens(tokenizer,row['problem'])
        pid=row['experiment_id']
        key=hashlib.sha256(pid.encode()).hexdigest()[:20]
        for run in range(args.runs):
            for policy,capacity,name in cells:
                path=out/name/f'{key}_r{run}.json'
                if path.exists():
                    prior=json.loads(path.read_text())
                    assert prior['problem_id']==pid and prior['run']==run
                    if prior.get('grading_errors'):
                        raise RuntimeError(f'Unresolved grading errors in {path}')
                    continue
                print(f'START {name} {pid} run={run} prompt={ids.shape[1]}',flush=True)
                base=dict(problem_id=pid, source_index=row['source_index'], run=run,
                          policy=policy, capacity=capacity, recent=64,
                          prompt_tokens=ids.shape[1], prompt_token_ids=ids[0].tolist(),
                          gold=row['answer'], level=row.get('level'), subject=row.get('subject'),
                          generation_seed=stable_seed(pid,run,'generation'),
                          eviction_seed=stable_seed(pid,run,'eviction'))
                if policy!='native' and ids.shape[1]>=capacity-64:
                    atomic_json(path,dict(base,status='budget_ineligible'))
                    print(f'INELIGIBLE {name} {pid}',flush=True)
                    continue
                pending=path.with_suffix('.ungraded')
                if pending.exists():
                    raw=json.loads(pending.read_text())
                    assert raw['problem_id']==pid and raw['run']==run
                    completion=raw['completion']
                    metrics=raw
                else:
                    # Empty between answers only; record allocated and reserved peaks.
                    torch.cuda.empty_cache()
                    started=time.time()
                    tokens,metrics=generate_one(model,ids,policy=policy,capacity=capacity,recent=64,
                        max_new_tokens=args.max_new_tokens,generation_seed=base['generation_seed'],
                        eviction_seed=base['eviction_seed'],sampling=sampling)
                    completion=tokenizer.decode(tokens,skip_special_tokens=False)
                    raw=dict(base,**metrics,status='generated',started_at=started,
                             token_ids=tokens,completion=completion)
                    atomic_json(pending,raw)
                graded=subprocess.run([sys.executable,'-m','kvcompress.qwen35.grade'],
                    input=json.dumps(dict(completion=completion,gold=row['answer'],termination=metrics['termination'])),
                    text=True,capture_output=True,timeout=30,cwd=ROOT,check=True)
                grade=json.loads(graded.stdout)
                atomic_json(path,dict(raw,**grade,status='complete'))
                path.with_suffix('.ungraded').unlink()
                if grade['grading_errors']:
                    raise RuntimeError(f"Grading error persisted in {path}: {grade['grading_errors']}")
                print(f"DONE {name} {pid} correct={grade['final_correct']} paper={grade['paper_correct']} "
                      f"tokens={metrics['generated_tokens']} end={metrics['termination']} "
                      f"tok_s={metrics['output_tok_s']:.2f} evictions={metrics['eviction_layer_events']}",flush=True)
                summarize(out,[x[2] for x in cells],len(rows)*args.runs)
    for h in hooks:
        h.remove()
    print('FINISHED',json.dumps(summarize(out,[x[2] for x in cells],len(rows)*args.runs)),flush=True)


if __name__=='__main__':
    main()
