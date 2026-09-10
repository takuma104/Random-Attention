"""CPU-only final analysis following qwen35-main32k-protocol.md.

Refuses incomplete or differently configured runs. Interim reports use the
separate audit_progress.py and do not run these confirmatory decisions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.qwen35.analyze_math import analyze,audit_and_load,paired_stats

SELECTORS=['random_pp_C1024 - recency_pp_C1024','random_pp_C1024 - snapkv_pp_C1024']


def holm_adjust(pvalues):
    order=sorted(pvalues,key=pvalues.get)
    adjusted={}
    previous=0.
    for rank,key in enumerate(order):
        p=float(pvalues[key])
        if not 0<=p<=1:
            raise ValueError('Invalid p-value')
        previous=min(1.,max(previous,(len(order)-rank)*p))
        adjusted[key]=previous
    return adjusted


def primary_decisions(contrasts):
    preservation=contrasts['random_pp_C2048 - native']['final_correct']
    adjusted=holm_adjust({key:contrasts[key]['final_correct']['exact_sign_p'] for key in SELECTORS})
    selectors={}
    for key in SELECTORS:
        stats=contrasts[key]['final_correct']
        low,high=stats['ci95']
        direction='random' if low>0 else 'comparator' if high<0 else 'unresolved'
        selectors[key]=dict(stats,holm_p=adjusted[key],
            supported_direction=direction if adjusted[key]<.05 else 'unresolved',
            family_size=2,ci_multiplicity_adjusted=False)
    return dict(accuracy_preservation=dict(preservation,margin=-.02,
                    criterion='Two-sided 95% paired problem-bootstrap CI lower bound > -0.02',
                    supports_target=preservation['ci95'][0]>-.02),selector_comparisons=selectors)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    manifest,records=audit_and_load(args.root)
    settings=manifest['settings']
    assert len(settings['problem_ids'])==500 and settings['args']['runs']==2
    assert settings['batch_size']==2 and settings['args']['max_new_tokens']==32768
    assert set(records)=={'native','random_pp_C1024','random_pp_C2048','recency_pp_C1024','snapkv_pp_C1024'}
    # Reuse the descriptive/paired baseline analysis and add the SnapKV contrast.
    snap={field:paired_stats(records['random_pp_C1024'],records['snapkv_pp_C1024'],field)
          for field in ('final_correct','paper_correct')}
    terminated_accuracy={name:sum(r['final_correct_terminated'] for r in cell.values())/len(cell)
                         for name,cell in records.items()}
    del records
    report=analyze(args.root)
    report['paired_contrasts'][SELECTORS[1]]=snap
    for name,value in terminated_accuracy.items():
        report['cells'][name]['eos_required_accuracy']=value
    report['primary_decisions']=primary_decisions(report['paired_contrasts'])
    report['analysis_protocol']='docs/plans/qwen35-main32k-protocol.md'
    report['original_manifest_sha256']=hashlib.sha256((args.root/'manifest.json').read_bytes()).hexdigest()
    report['notes']=[
        'Final preregistered B2 main analysis; no B1 pilot scores pooled.',
        'Problem-clustered percentile bootstrap, 10000 replicates, seed20260910; marginal 95% CIs.',
        'Holm adjustment applies ONLY to the two preregistered C1024 selector sign tests.',
        'Accuracy-preservation criterion applies ONLY to Random C2048 vs native.',
        'All other contrasts and paper-compatible metrics are secondary/exploratory.',
        'Non-significance is not equivalence. Caps remain in the denominator.',
        'Generation tok/s and shared batch peak memory are not iso-workload serving benchmarks.',
    ]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['primary_decisions'],indent=2))


if __name__=='__main__':
    main()
