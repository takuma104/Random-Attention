"""Post-hoc descriptive SAME-question legacy-score comparison; no causal test."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
from scripts.qwen3_control.analyze_math import ROOT,audit,digest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path); args=ap.parse_args()
    control=ROOT/'results/qwen3_control/pilot32k_b2_v1'
    settings,_,_=audit(control); ids=settings['problem_ids']
    report=dict(scope='Post-hoc descriptive matched-question subset; legacy scores only; no cross-model causal inference or hypothesis tests',
                problem_ids=ids,models={},source_sha256=digest(Path(__file__)))
    for name,relative in [('Qwen3-4B','results/qwen3_control/pilot32k_b2_v1'),('Qwen3.5-4B','results/qwen35/main32k_b2_v3')]:
        root=ROOT/relative; manifest=json.loads((root/'manifest.json').read_text()); s=manifest['settings']
        assert s['batch_size']==2 and s['args']['max_new_tokens']==32768
        assert s['artifacts']['model_id']=='Qwen/'+name
        assert s['artifacts']['dataset_sha256']==settings['artifacts']['dataset_sha256']
        for path,sha in s['source_hashes'].items(): assert digest(ROOT/path)==sha,path
        cells={}; h=hashlib.sha256()
        for cell in ('native','random_pp_C1024','random_pp_C2048'):
            rows=[]
            for pid in ids:
                for run in (0,1):
                    rel=Path(cell)/(hashlib.sha256(pid.encode()).hexdigest()[:20]+f'_r{run}.json')
                    data=(root/rel).read_bytes(); h.update(str(rel).encode()); h.update(data)
                    r=json.loads(data)
                    assert r['problem_id']==pid and r['run']==run and r['status']=='complete' and not r['grading_errors']
                    rows.append(r)
            cells[cell]=dict(n=len(rows),legacy_final_accuracy=sum(r['final_correct'] for r in rows)/len(rows),
                eos_required_accuracy=sum(r['final_correct_terminated'] for r in rows)/len(rows),
                cap_rate=sum(r['termination']=='length' for r in rows)/len(rows),
                mean_tokens=sum(r['generated_tokens'] for r in rows)/len(rows),
                eviction_exposure=sum(r['eviction_layer_events']>0 for r in rows)/len(rows))
        report['models'][name]=dict(root=relative,manifest_sha256=digest(root/'manifest.json'),
            subset_data_sha256=h.hexdigest(),sampling=s['sampling'],cells=cells)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['models'],indent=2))


if __name__=='__main__': main()
