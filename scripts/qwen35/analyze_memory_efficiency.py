"""CPU-only memory-based batch selection and cross-batch decode comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.qwen35.analyze_efficiency import analyze,distribution,GIB

METHODS=('random_pp_C1024','random_pp_C2048','recency_pp_C1024','snapkv_pp_C1024')


def choose_batch(points,budget):
    eligible=[p for p in points if .95<=p['peak_allocated_gib']/budget<=1.]
    if eligible: return max(eligible,key=lambda p:p['batch']),True
    below=[p for p in points if p['peak_allocated_gib']<=budget]
    assert below,'Need a smaller calibration batch'
    chosen=max(below,key=lambda p:p['batch'])
    assert any(p['batch']==chosen['batch']+8 and p['peak_allocated_gib']>budget for p in points), 'Need adjacent larger calibration'
    return chosen,False


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def select(calibration_root,reference_root):
    reference=analyze(reference_root)
    budget=max(r['peak_allocated_gib'] for r in reference['summaries']
        if r['batch']==8 and r['method']=='native_preallocated' and r['nominal_context']==32768)
    source_input=json.loads((reference_root/'input.json').read_text())
    ref_settings=json.loads((reference_root/'manifest.json').read_text())['settings']
    points=[]
    for path in sorted(calibration_root.iterdir()):
        if not path.is_dir(): continue
        report=analyze(path); run=report['runs'][0]
        assert report['n_cells']==1 and report['n_windows']==2
        settings=json.loads((path/'manifest.json').read_text())['settings']
        for field in ('source_hashes','artifacts','torch','kernels'):
            assert settings[field]==ref_settings[field]
        assert settings['args']['steps']==128 and settings['args']['repeats']==1
        assert settings['args']['modes']==ref_settings['args']['modes']
        inputs=json.loads((path/'input.json').read_text())
        assert inputs['prompt_ids']==source_input['prompt_ids']
        assert inputs['forced_ids']==source_input['forced_ids'][:len(inputs['forced_ids'])]
        assert {s['nominal_context'] for s in report['summaries']}=={4096}
        points.append(dict(method=run['method'],batch=run['batch'],path=str(path),
            data_sha256=report['completed_data_sha256'],
            peak_allocated_gib=max(s['peak_allocated_gib'] for s in report['summaries']),
            whole_run_peak_allocated_gib=run['overall_peak_allocated_bytes']/GIB))
    assert len({(p['method'],p['batch']) for p in points})==len(points)
    assert {p['method'] for p in points}==set(METHODS)
    selected=[]
    for method in METHODS:
        point,matched=choose_batch([p for p in points if p['method']==method],budget)
        selected.append(dict(method=method,batch=point['batch'],calibration_fraction=point['peak_allocated_gib']/budget,
            calibration_within_95_to_100_percent=matched,
            measurement_subdir=f'{method}_B{point["batch"]}',
            reason='Within prespecified interval' if matched else 'Adjacent batch exceeds budget; closest under-budget grid point, NOT within tolerance'))
    return dict(calibration_root=str(calibration_root),reference_root=str(reference_root),reference_data_sha256=reference['completed_data_sha256'],
        budget_decode_allocated_gib=budget,calibrations=points,selected=selected,
        rule='Memory-only, batch multiples of 8, target 95-100% of native_preallocated B8 decode peak allocated; disclose fallback outside interval')


def paired_metrics(rows,reference):
    key=lambda r:(r['nominal_context'],r['repeat'],r['mode'])
    a={key(r):r for r in rows}; b={key(r):r for r in reference}
    assert len(a)==len(rows) and len(b)==len(reference) and set(a)==set(b)
    for k in a:
        for field in ('logical_start','logical_end','steps'):
            assert a[k][field]==b[k][field],f'Unmatched {field}'
    result=[]
    for mode in ('throughput','synchronized'):
        keys=sorted(k for k in a if k[2]==mode)
        result.append(dict(mode=mode,
            aggregate_tokens_per_second=distribution([a[k]['tokens_per_second'] for k in keys]),
            ms_per_step=distribution([a[k]['ms_per_step'] for k in keys]),
            paired_aggregate_throughput_ratio=distribution([a[k]['tokens_per_second']/b[k]['tokens_per_second'] for k in keys]),
            paired_step_latency_ratio=distribution([a[k]['ms_per_step']/b[k]['ms_per_step'] for k in keys])))
    return result


def compare(root,selection_path):
    selection=json.loads(selection_path.read_text()); ref_root=Path(selection['reference_root'])
    assert select(Path(selection['calibration_root']),ref_root)==selection,'Calibration or selection changed'
    assert {p.name for p in root.iterdir() if p.is_dir()}=={p['measurement_subdir'] for p in selection['selected']}
    ref=analyze(ref_root); assert ref['completed_data_sha256']==selection['reference_data_sha256']
    ref_settings=json.loads((ref_root/'manifest.json').read_text())['settings']
    reference=json.loads((ref_root/'B8/native_preallocated.json').read_text())
    ref_rows=[r for r in reference['records'] if r['nominal_context']==32768]
    budget=max(r['peak_allocated_bytes'] for r in ref_rows)/GIB
    assert budget==selection['budget_decode_allocated_gib']
    comparisons=[]; total_hours=0
    for chosen in selection['selected']:
        path=root/chosen['measurement_subdir']; report=analyze(path)
        settings=json.loads((path/'manifest.json').read_text())['settings']
        for field in ('source_hashes','artifacts','torch','kernels','maximum_logical_length'):
            assert settings[field]==ref_settings[field],f'Mismatch: {field}'
        for field in ('steps','repeats','modes'):
            assert settings['args'][field]==ref_settings['args'][field]
        assert settings['args']['contexts']=='32768'
        assert report['input_token_sha256']==ref['input_token_sha256']
        assert report['n_cells']==1 and report['n_windows']==6
        run=report['runs'][0]
        assert run['method']==chosen['method'] and run['batch']==chosen['batch']
        raw=json.loads((path/f'B{run["batch"]}'/f'{run["method"]}.json').read_text())
        peak=max(r['peak_allocated_bytes'] for r in raw['records'])/GIB
        comparisons.append(dict(method=run['method'],batch=run['batch'],data_sha256=report['completed_data_sha256'],
            peak_decode_allocated_gib=peak,fraction_of_reference_budget=peak/budget,
            within_95_to_100_percent=.95<=peak/budget<=1.,
            peak_decode_reserved_gib=max(r['peak_reserved_bytes'] for r in raw['records'])/GIB,
            whole_run_peak_allocated_gib=run['overall_peak_allocated_bytes']/GIB,
            gpu_before=run['gpu_before'],gpu_after=run['gpu_after'],
            paired_metrics=paired_metrics(raw['records'],ref_rows)))
        total_hours+=report['total_run_wall_hours']
    return dict(audit='passed',selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        reference_data_sha256=ref['completed_data_sha256'],input_token_sha256=ref['input_token_sha256'],
        total_compressed_run_wall_hours=total_hours,
        reference=dict(method='native_preallocated',batch=8,peak_decode_allocated_gib=budget,
            peak_decode_reserved_gib=max(r['peak_reserved_bytes'] for r in ref_rows)/GIB,
            whole_run_peak_allocated_gib=reference['overall_peak_allocated_bytes']/GIB,
            metrics=paired_metrics(ref_rows,ref_rows)),comparisons=comparisons,
        notes=['Only decode PyTorch peak allocated is budgeted; reserved, NVML and prefill peak are not matched.',
               'Batch was selected from calibration memory, not speed or accuracy; fallback outside 95-100% must be labeled.',
               'Same trace and logical windows, different batches; repeated identical inputs are not independent serving requests.',
               'Model+LM head only. No sampling/EOS/scheduler. Adjacent windows give descriptive ranges, not confidence intervals.',
               'High-batch accuracy is untested; substantial B2 MATH accuracy loss prevents claims of quality-preserving speedup.'])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['select','compare'])
    ap.add_argument('--root',required=True,type=Path); ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--reference',default='results/qwen35/efficiency_v1',type=Path)
    ap.add_argument('--selection',type=Path)
    args=ap.parse_args()
    report=select(args.root,args.reference) if args.action=='select' else compare(args.root,args.selection)
    save(args.out,report)
    print(json.dumps(report.get('selected',report.get('comparisons')),indent=2))


if __name__=='__main__': main()
