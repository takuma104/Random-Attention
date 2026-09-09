"""Audit completed experiment cells and compute paired problem-clustered CIs.

No GPU/model imports. All statistics are exploratory for the 50-question pilot.
Bootstrap resamples PROBLEMS, preserving all runs of a problem. No truncated or
incorrect examples are filtered. Optional reference exposure uses native lengths.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest


def audit_and_load(root):
    manifest=json.loads((root/'manifest.json').read_text())
    settings=manifest['settings']
    problems=settings['problem_ids']
    runs=settings['args']['runs']
    cap=settings['args']['max_new_tokens']
    expected={(p,r) for p in problems for r in range(runs)}
    records={}
    cell_names=[]
    for spec in settings['args']['cells'].split(','):
        policy,_,capacity=spec.partition(':')
        cell_names.append('native' if policy=='native' else f'{policy}_C{capacity}')
    for name in cell_names:
        cell={}
        for path in sorted((root/name).glob('*.json')):
            row=json.loads(path.read_text())
            key=(row['problem_id'],row['run'])
            assert key not in cell, f'Duplicate {name} {key}'
            assert row['status']=='complete', f'Incomplete/ineligible {path}'
            assert not row['grading_errors'], f'Grade errors {path}'
            ids=row['token_ids']
            assert len(ids)==row['generated_tokens']
            assert row['prompt_tokens']==len(row['prompt_token_ids'])
            assert row['logical_cached_tokens']==row['prompt_tokens']+len(ids)-1
            assert not any(t in row['eos_ids'] for t in ids[:-1]), f'Early EOS ignored {path}'
            if row['termination']=='eos':
                assert ids[-1] in row['eos_ids']
            else:
                assert row['termination']=='length' and len(ids)==cap
                assert ids[-1] not in row['eos_ids']
            if name!='native':
                assert row['prompt_tokens']<row['capacity']-row['recent']
                events=max(0,(row['logical_cached_tokens']-row['capacity'])//row['recent'])
                assert row['eviction_layer_events']==8*events
                assert all(n<=row['capacity']+row['recent'] for n in row['final_physical_lengths'])
            else:
                assert row['eviction_layer_events']==0
            cell[key]=row
        assert set(cell)==expected, f'Incomplete cell {name}: {len(cell)}/{len(expected)}'
        records[name]=cell
    # Paired prompts and sampling seeds must agree. No equality requirement for
    # output length, since eviction may change termination.
    reference=records['native'] if 'native' in records else records[cell_names[0]]
    for cell in records.values():
        for key,row in cell.items():
            for field in ('prompt_token_ids','generation_seed','eviction_seed','sampling','gold','eos_ids'):
                assert row[field]==reference[key][field], f'Paired {field} mismatch: {key}'
    return manifest,records


def paired_stats(a,b,field,replicates=10000):
    problem_diffs=defaultdict(list)
    for key in a:
        problem_diffs[key[0]].append(float(a[key][field])-float(b[key][field]))
    diffs=np.array([np.mean(problem_diffs[p]) for p in sorted(problem_diffs)])
    rng=np.random.default_rng(20260910)
    boot=diffs[rng.integers(len(diffs),size=(replicates,len(diffs)))].mean(1)
    wins=int((diffs>0).sum()); losses=int((diffs<0).sum())
    return dict(n_problems=len(diffs), difference=float(diffs.mean()),
                ci95=[float(x) for x in np.quantile(boot,[.025,.975])],
                problem_wins=wins,problem_losses=losses,problem_ties=int((diffs==0).sum()),
                exact_sign_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
                bootstrap_replicates=replicates)


def analyze(root):
    manifest,records=audit_and_load(root)
    cells={}
    for name,cell in records.items():
        rows=list(cell.values()); n=len(rows)
        mean=lambda field: float(np.mean([r[field] for r in rows]))
        lengths=np.array([r['generated_tokens'] for r in rows])
        ended=[r for r in rows if r['termination']=='eos']
        cells[name]=dict(n=n,final_accuracy=mean('final_correct'),paper_accuracy=mean('paper_correct'),
                         cap_rate=float(np.mean([r['termination']=='length' for r in rows])),
                         thinking_unclosed_rate=float(np.mean([not r['thinking_closed'] for r in rows])),
                         n_eos=len(ended),n_eos_incorrect=sum(not r['final_correct'] for r in ended),
                         mean_tokens=float(lengths.mean()),median_tokens=float(np.median(lengths)),
                         p90_tokens=float(np.quantile(lengths,.9)),
                         eviction_exposure=float(np.mean([r['eviction_layer_events']>0 for r in rows])),
                         generation_seconds=sum(r['elapsed_seconds'] for r in rows),
                         aggregate_tok_s=float(lengths.sum()/sum(r['elapsed_seconds'] for r in rows)),
                         peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
                         mean_kv_mib=float(np.mean([r['cache_bytes']['kv'] for r in rows]))/2**20,
                         recurrent_mib=rows[0]['cache_bytes']['recurrent']/2**20,
                         conv_mib=rows[0]['cache_bytes']['conv']/2**20)
    contrasts={}
    if 'native' in records:
        for name in records:
            if name!='native':
                contrasts[f'{name} - native']={field:paired_stats(records[name],records['native'],field)
                                               for field in ('final_correct','paper_correct')}
    for name in records:
        if name.startswith('random_pp_'):
            other=name.replace('random_pp_','recency_pp_')
            if other in records:
                contrasts[f'{name} - {other}']={field:paired_stats(records[name],records[other],field)
                                               for field in ('final_correct','paper_correct')}
    files=sorted(root.glob('*/*.json'))
    aggregate_hash=hashlib.sha256()
    for p in files:
        aggregate_hash.update(str(p.relative_to(root)).encode())
        aggregate_hash.update(p.read_bytes())
    return dict(audit='passed',input_path=str(root),manifest_commit=manifest['git_commit'],
                completed_data_sha256=aggregate_hash.hexdigest(),n_problems=len(manifest['settings']['problem_ids']),
                runs_per_problem=manifest['settings']['args']['runs'],
                max_new_tokens=manifest['settings']['args']['max_new_tokens'],
                total_generation_hours=sum(c['generation_seconds'] for c in cells.values())/3600,
                cells=cells,paired_contrasts=contrasts,
                notes=['Exploratory problem-clustered percentile bootstrap, seed20260910; no multiplicity correction.',
                       'Non-significance is not equivalence. Pilot selection does not isolate architectural causality.',
                       'Cap examples remain in denominator; EOS-conditioned accuracy is a diagnostic, not primary.',
                       'Per-answer speed is not iso-workload or iso-allocator efficiency.'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    report=analyze(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(f"Audit passed; generation {report['total_generation_hours']:.2f}h")
    for name,c in report['cells'].items():
        print(f"{name:24} final={c['final_accuracy']:.1%} paper={c['paper_accuracy']:.1%} "
              f"cap={c['cap_rate']:.1%} EOSwrong={c['n_eos_incorrect']}/{c['n_eos']} "
              f"mean_tokens={c['mean_tokens']:.0f} eviction={c['eviction_exposure']:.1%}")
    for name,metrics in report['paired_contrasts'].items():
        c=metrics['final_correct']
        print(name, 'diff_pp=',round(c['difference']*100,2),'CI_pp=',[round(x*100,2) for x in c['ci95']],
              'wins/losses=',c['problem_wins'],c['problem_losses'],'sign_p=',c['exact_sign_p'])


if __name__=='__main__':
    main()
