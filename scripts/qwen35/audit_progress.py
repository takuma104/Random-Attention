"""CPU-only audit of a fixed completed prefix; never mutate a live run.

No interim hypothesis tests or sample-size adaptation. Uses a temporary derived
snapshot solely to reuse the strict completed-cell auditor. The original manifest
hash and exact selected problem IDs are retained in the operational report.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
from datetime import datetime,timezone
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.qwen35.analyze_math import audit_and_load

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--problems',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    original=(args.root/'manifest.json').read_bytes()
    manifest=json.loads(original)
    settings=manifest['settings']
    all_ids=settings['problem_ids']
    if not 0<args.problems<=len(all_ids):
        raise ValueError('Invalid prefix length')
    selected=all_ids[:args.problems]
    for name,digest in settings['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Frozen source changed: {name}'
    aggregate=hashlib.sha256()
    rows_by_cell={}
    total_bytes=0
    with tempfile.TemporaryDirectory(prefix='audit-qwen35-') as temp:
        target=Path(temp)
        derived=json.loads(original)
        derived['settings']['problem_ids']=selected
        derived['settings']['args']['limit']=len(selected)
        (target/'manifest.json').write_text(json.dumps(derived))
        for spec in settings['args']['cells'].split(','):
            policy,_,cap=spec.partition(':')
            cell='native' if policy=='native' else f'{policy}_C{cap}'
            (target/cell).mkdir()
            for pid in selected:
                key=hashlib.sha256(pid.encode()).hexdigest()[:20]
                for run in range(settings['args']['runs']):
                    relative=Path(cell)/f'{key}_r{run}.json'
                    raw=(args.root/relative).read_bytes()
                    aggregate.update(str(relative).encode()); aggregate.update(raw)
                    total_bytes+=len(raw)
                    (target/relative).write_bytes(raw)
        _,records=audit_and_load(target)
        for cell,values in records.items():
            rows=list(values.values())
            groups={r['batch_id']:[] for r in rows}
            for r in rows:
                groups[r['batch_id']].append(r)
            seconds=sum(r['elapsed_seconds'] for r in rows)
            useful=sum(r['generated_tokens'] for r in rows)
            scheduled=sum(len(g)*max(r['generated_tokens'] for r in g) for g in groups.values())
            rows_by_cell[cell]=dict(n=len(rows),n_final_correct=sum(r['final_correct'] for r in rows),
                n_cap=sum(r['termination']=='length' for r in rows),
                n_eos_wrong=sum(r['termination']=='eos' and not r['final_correct'] for r in rows),
                mean_tokens=useful/len(rows),generation_hours=seconds/3600,
                aggregate_useful_tok_s=useful/seconds,useful_row_step_fraction=useful/scheduled,
                peak_shared_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
                subjects=dict(Counter(r['subject'] for r in rows)))
    hours=sum(c['generation_hours'] for c in rows_by_cell.values())
    report=dict(status='passed',interim=True,scope='Operational monitoring only; no hypothesis tests.',
        captured_at=datetime.now(timezone.utc).isoformat(),input_path=str(args.root),
        original_manifest_sha256=hashlib.sha256(original).hexdigest(),
        prefix_data_sha256=aggregate.hexdigest(),selected_problem_ids=selected,
        n_problems=len(selected),n_planned_problems=len(all_ids),
        total_generation_hours=hours,linear_full_run_days=hours/len(selected)*len(all_ids)/24,
        raw_answer_bytes=total_bytes,frozen_source_hashes_match=True,cells=rows_by_cell,
        caveats=['Deterministic dataset prefix, not a new random sample; runtime extrapolation is provisional.',
                 'No change to sample size, methods, budgets, or sampling from interim accuracy.',
                 'Shared peak memory is batch-wide; elapsed_seconds is allocated batch time/2.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
