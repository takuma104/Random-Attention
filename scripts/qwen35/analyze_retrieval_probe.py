"""CPU-only strict probe audit and case-clustered descriptive/paired analysis."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[2]


def clustered_difference(differences):
    d=np.asarray(differences,dtype=float)
    assert len(d)>0 and np.isfinite(d).all()
    rng=np.random.default_rng(20260916)
    boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
    wins=int((d>0).sum()); losses=int((d<0).sum())
    return dict(n_cases=len(d),difference=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        case_wins=wins,case_losses=losses,case_ties=int((d==0).sum()),
        exact_sign_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
        bootstrap_replicates=10000,bootstrap_seed=20260916)


def audit(root):
    manifest=json.loads((root/'manifest.json').read_text()); settings=manifest['settings']; args=settings['args']
    for name,digest in settings['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Source changed: {name}'
    gaps=[int(x) for x in args['gaps'].split(',')]
    cases=[f'{args["phase"]}-{i:03d}' for i in range(args['cases'])]
    cells={}
    for spec in args['cells'].split(','):
        policy,_,capacity=spec.partition(':'); cap=int(capacity or 0)
        name=policy if policy=='native' else f'{policy}_C{cap}'
        rows={}
        for path in sorted((root/name).glob('*.json')):
            r=json.loads(path.read_text()); key=(r['case_id'],r['gap_tokens'])
            assert key not in rows and r['phase']==args['phase'] and r['batch_size']==2
            assert r['policy']==policy and r['capacity']==cap and len(set(r['eviction_seeds']))==2
            p=r['prompt_length']; f0,f1=r['fact_span']; v0,v1=r['value_span']
            assert p==len(r['prompt_ids']) and f0==p and p<=v0<v1<=f1
            assert len(r['trace_ids'])==f1-f0+r['gap_tokens']+r['query_length']
            assert hashlib.sha256(json.dumps(r['trace_ids']).encode()).hexdigest()==r['trace_sha256']
            final=p+len(r['trace_ids'])
            assert r['final_logical_length']==final and r['snapshot_logical_length']==final-1
            before_events=0 if policy=='native' else 8*max(0,(final-1-cap)//64)
            after_events=0 if policy=='native' else 8*max(0,(final-cap)//64)
            assert r['eviction_layer_events_before_last_query']==before_events
            assert r['eviction_layer_events']==after_events
            scores=np.array(r['label_logits']); probs=np.array(r['conditional_probabilities'])
            assert scores.shape==probs.shape==(2,4) and np.isfinite(scores).all()
            e=np.exp(scores-scores.max(1,keepdims=True))
            np.testing.assert_allclose(probs,e/e.sum(1,keepdims=True),rtol=1e-6,atol=1e-7)
            assert r['predictions']==scores.argmax(1).tolist()
            assert 0<=r['target']<4 and r['correct']==[x==r['target'] for x in r['predictions']]
            mass=np.array(r['label_probability_mass']); assert mass.shape==(2,) and ((mass>=0)&(mass<=1.000001)).all()
            fact=np.array(r['fact_token_counts']); value=np.array(r['value_token_counts'])
            assert fact.shape==value.shape==(2,8,4)
            assert ((fact>=0)&(fact<=f1-f0)).all() and ((value>=0)&(value<=v1-v0)).all()
            assert (value<=fact).all()
            overlap=np.array(r['older_generated_head_jaccard_by_layer'])
            assert overlap.shape==(2,8) and ((overlap>=0)&(overlap<=1)).all()
            if policy in ('native','recency_pp','random_shared_pp'):
                np.testing.assert_array_equal(overlap,np.ones((2,8)))
            if policy=='native' or before_events==0:
                np.testing.assert_array_equal(fact,np.full((2,8,4),f1-f0))
                np.testing.assert_array_equal(value,np.full((2,8,4),v1-v0))
            if policy=='native':
                np.testing.assert_array_equal(scores[0],scores[1])
            if policy=='recency_pp' and final-1-64>=cap and f1<=final-1-(cap-p+64):
                # Once even the oldest possible recent generated tail is later
                # than the fact, no fact positions may survive strict recency.
                assert not fact.any()
            assert r['elapsed_seconds']>0
            rows[key]=r
        assert set(rows)=={(c,g) for c in cases for g in gaps},f'Incomplete {name}'
        cells[name]=rows
    reference=cells.get('native',next(iter(cells.values())))
    for rows in cells.values():
        for key,r in rows.items():
            ref=reference[key]
            for field in ('prompt_ids','trace_sha256','fact_span','value_span','target','label_token_ids','eviction_seeds','options'):
                assert r[field]==ref[field],f'Paired {field} mismatch'
            if 'native' in cells and r['eviction_layer_events']==0:
                np.testing.assert_array_equal(r['label_logits'],ref['label_logits'])
    return manifest,cells,cases,gaps


def analyze(root):
    manifest,cells,cases,gaps=audit(root)
    summary={}
    for name,rows in cells.items():
        summary[name]={}
        for gap in gaps:
            selected=[rows[(c,gap)] for c in cases]
            probs=np.array([r['conditional_probabilities'] for r in selected])
            target_probs=np.array([probs[i,:,r['target']] for i,r in enumerate(selected)])
            correct=np.array([r['correct'] for r in selected],dtype=float)
            complete_values=np.array([np.array(r['value_token_counts'])==(r['value_span'][1]-r['value_span'][0]) for r in selected])
            summary[name][str(gap)]=dict(n_independent_cases=len(cases),n_row_readouts=2*len(cases),
                accuracy=float(correct.mean()),mean_target_probability=float(target_probs.mean()),
                mean_label_mass=float(np.mean([r['label_probability_mass'] for r in selected])),
                any_full_value_head_rate=float(complete_values.any(axis=(2,3)).mean()),
                full_value_head_fraction=float(complete_values.mean()),
                any_fact_token_rate=float(np.mean([np.array(r['fact_token_counts']).reshape(2,-1).any(1) for r in selected])),
                mean_older_head_jaccard=float(np.mean([r['older_generated_head_jaccard_by_layer'] for r in selected])),
                generation_seconds=sum(r['elapsed_seconds'] for r in selected))
    contrasts={}
    for other in ('random_shared_pp_C1024','recency_pp_C1024','snapkv_pp_C1024','native'):
        if 'random_pp_C1024' not in cells or other not in cells: continue
        label=f'random_pp_C1024 - {other}'
        contrasts[label]={}
        for gap in gaps:
            diffs=[np.mean(cells['random_pp_C1024'][(c,gap)]['correct'])-np.mean(cells[other][(c,gap)]['correct']) for c in cases]
            contrasts[label][str(gap)]=clustered_difference(diffs)
    far=[2048,8192,16384]
    primary=None
    if {'random_pp_C1024','random_shared_pp_C1024'}<=set(cells) and set(far)<=set(gaps):
        diffs=[np.mean([np.mean(cells['random_pp_C1024'][(c,g)]['correct'])-np.mean(cells['random_shared_pp_C1024'][(c,g)]['correct']) for g in far]) for c in cases]
        primary=dict(contrast='Random1024 - Random-shared1024; accuracy averaged over gaps2048/8192/16384 within each case',
                     **clustered_difference(diffs))
    calibration_gate=None
    if manifest['settings']['args']['phase']=='calibration' and len(cases)==8 and {0,8192,16384}<=set(gaps) and 'native' in cells:
        calibration_gate={str(g):sum(cells['native'][(c,g)]['correct'][0] for c in cases)>=7 for g in (0,8192,16384)}
    h=hashlib.sha256()
    for path in sorted(root.glob('*/*.json')):
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return dict(audit='passed',input_path=str(root),manifest_commit=manifest['git_commit'],
        completed_data_sha256=h.hexdigest(),n_cases=len(cases),gaps=gaps,
        total_generation_hours=sum(r['elapsed_seconds'] for rows in cells.values() for r in rows.values())/3600,
        cells=summary,paired_case_contrasts=contrasts,primary_diversity_contrast=primary,
        native_calibration_gate=calibration_gate,
        notes=['Cases, not heads or duplicate rows, are the independent units. Paired bootstrap seed20260916,10000 replicates.',
               'Gap-specific contrasts are exploratory; small samples and saturated outcomes can give degenerate bootstrap CIs.',
               'Retained source positions are measured before the final query token; later-token KV or DeltaNet can already carry the fact.',
               'Teacher-forced lookup is not free-generation MATH accuracy or a causal architecture comparison.'])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); report=analyze(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('Audit',report['audit'],'hours',report['total_generation_hours'],'calibration_gate',report['native_calibration_gate'])
    for cell,gaps in report['cells'].items():
        for gap,r in gaps.items():
            print(cell,'gap',gap,'accuracy',r['accuracy'],'p_target',round(r['mean_target_probability'],4),'value_any_head',r['any_full_value_head_rate'],'Jaccard',round(r['mean_older_head_jaccard'],4))


if __name__=='__main__': main()
