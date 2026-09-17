"""CPU-only, post-main capacity extension. Never rewrites raw scores or data."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.qwen35.analyze_math import audit_and_load

ROOT=Path(__file__).resolve().parents[2]
BASELINE_SHA='b3ad6d1939371a128587508e329ab92c2b3d09fa7d84a74ba07b41d056d26a66'
CELLS={'random_pp_C4096':4096,'random_pp_C8192':8192}


def data_hash(root):
    h=hashlib.sha256()
    for path in sorted(root.glob('*/*.json')):
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def frozen_sources(settings):
    for name,digest in settings['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Frozen source changed: {name}'


def check_settings(settings,reference):
    for field in ('batch_size','artifacts','sampling','source_hashes','torch','transformers','python','kernels','problem_ids','batch_policy','timing_policy'):
        assert settings[field]==reference[field],f'Reference settings mismatch: {field}'
    assert set(settings['args'])==set(reference['args'])
    for field in set(settings['args'])-{'out','cells'}:
        assert settings['args'][field]==reference['args'][field],f'Reference argument mismatch: {field}'
    assert settings['batch_size']==2 and settings['args']['runs']==2 and settings['args']['max_new_tokens']==32768


def check_prefix(row,native):
    for field in ('problem_id','run','prompt_token_ids','generation_seed','eviction_seed','sampling','gold','eos_ids'):
        assert row[field]==native[field],f'Paired {field} mismatch'
    # First eviction is AFTER attention at logical C+r. That forward's sampled
    # token also still belongs to the uncompressed prefix (prefill yields token0).
    length=min(len(native['token_ids']),row['capacity']+row['recent']-row['prompt_tokens']+1)
    assert length>0 and len(row['token_ids'])>=length
    assert row['token_ids'][:length]==native['token_ids'][:length],f'Pre-eviction prefix mismatch: {row["problem_id"]} run{row["run"]}'
    return length


def bootstrap_contrast(a,b,field,replicates=10000):
    assert set(a)==set(b)
    differences=defaultdict(list)
    for key in a: differences[key[0]].append(float(a[key][field])-float(b[key][field]))
    assert all(len(v)==2 for v in differences.values())
    d=np.array([np.mean(differences[p]) for p in sorted(differences)])
    rng=np.random.default_rng(20260917)
    boot=d[rng.integers(len(d),size=(replicates,len(d)))].mean(1)
    return dict(n_problems=len(d),difference=float(d.mean()),
        ci95=[float(v) for v in np.quantile(boot,[.025,.975])],
        ci97_5=[float(v) for v in np.quantile(boot,[.0125,.9875])],
        wins=int((d>0).sum()),losses=int((d<0).sum()),ties=int((d==0).sum()),
        bootstrap_replicates=replicates,bootstrap_seed=20260917)


def preservation(stats):
    return dict(**stats,family_size=2,margin=-.02,supports_target=stats['ci97_5'][0]>-.02,
        criterion='Individual 97.5% two-sided percentile CI lower > -0.02; Bonferroni new two-contrast family')


def describe(cell):
    rows=list(cell.values()); mean=lambda values:float(np.mean(values))
    seconds=sum(r['elapsed_seconds'] for r in rows)
    return dict(n=len(rows),final_accuracy=mean([r['final_correct'] for r in rows]),
        paper_accuracy=mean([r['paper_correct'] for r in rows]),
        eos_required_accuracy=mean([r['final_correct_terminated'] for r in rows]),
        cap_rate=mean([r['termination']=='length' for r in rows]),
        mean_tokens=mean([r['generated_tokens'] for r in rows]),
        median_tokens=float(np.median([r['generated_tokens'] for r in rows])),
        eviction_exposure=mean([r['eviction_layer_events']>0 for r in rows]),
        n_eos_wrong=sum(r['termination']=='eos' and not r['final_correct'] for r in rows),
        generation_hours=seconds/3600,aggregate_generation_tok_s=sum(r['generated_tokens'] for r in rows)/seconds,
        shared_peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30)


def load_reference(root):
    assert data_hash(root)==BASELINE_SHA,'Original main data changed'
    assert not list(root.rglob('*.batch-ungraded'))
    manifest,records=audit_and_load(root); frozen_sources(manifest['settings'])
    return manifest,records['native']


def analyze(root,reference_root):
    reference_manifest,native=load_reference(reference_root)
    manifest,records=audit_and_load(root); settings=manifest['settings']; ref=reference_manifest['settings']
    assert set(records)==set(CELLS) and len(settings['problem_ids'])==500
    assert not list(root.rglob('*.batch-ungraded'))
    assert len(list(root.glob('*/*.json')))==2000
    frozen_sources(settings)
    check_settings(settings,ref)
    prefix_tokens=0
    for name,cell in records.items():
        assert set(cell)==set(native)
        for key,row in cell.items():
            assert row['policy']=='random_pp' and row['capacity']==CELLS[name] and row['recent']==64
            prefix_tokens+=check_prefix(row,native[key])
    contrasts={name:{field:bootstrap_contrast(cell,native,field)
        for field in ('final_correct','paper_correct','final_correct_terminated')} for name,cell in records.items()}
    strata=[]
    for low,high in ((0,4096),(4096,8192),(8192,16384),(16384,32769)):
        keys=[k for k,r in native.items() if low<=r['generated_tokens']<high]
        strata.append(dict(native_generated_length_interval=[low,high],n_rows=len(keys),
            n_problems=len({k[0] for k in keys}),
            final_accuracy={name:float(np.mean([cell[k]['final_correct'] for k in keys])) if keys else None
                for name,cell in {'native':native,**records}.items()}))
    return dict(audit='passed',protocol='docs/plans/qwen35-capacity-frontier-protocol.md',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        completed_data_sha256=data_hash(root),reference_data_sha256=BASELINE_SHA,
        manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        reference_manifest_sha256=hashlib.sha256((reference_root/'manifest.json').read_bytes()).hexdigest(),
        matched_pre_eviction_prefixes=2000,matched_prefix_tokens=prefix_tokens,
        cells={name:describe(cell) for name,cell in {'native':native,**records}.items()},
        contrasts=contrasts,primary_decisions={name:preservation(stats['final_correct']) for name,stats in contrasts.items()},
        native_length_strata=strata,
        notes=['Prospectively fixed post-main extension; same MATH500 and already-known native data, not independent replication.',
               'Original C2048 preservation failure remains unchanged. New family has two comparisons only.',
               'Legacy raw scores are primary; separate formatting sensitivity must not rewrite these data.',
               '500 problem clusters retain two runs. Individual 97.5% bootstrap CIs use Bonferroni for nominal 95% family coverage.',
               'Bootstrap coverage is approximate; adjustment is not study-wide across all prior/adaptive experiments.',
               'All capped answers remain in denominator. Native-generated-length strata are descriptive row summaries only.',
               'Generation throughput/shared memory are not fixed-workload serving measurements.'])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path)
    ap.add_argument('--reference',default='results/qwen35/main32k_b2_v3',type=Path)
    ap.add_argument('--out',required=True,type=Path); args=ap.parse_args()
    report=analyze(args.root,args.reference)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['primary_decisions'],indent=2))


if __name__=='__main__': main()
