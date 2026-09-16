"""CPU-only audit of fixed decode windows, with descriptive paired speed ratios.

Three adjacent windows are NOT independent requests. No significance tests or
serving latency/QPS claims are made from these model-only measurements.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
MIB=2**20
GIB=2**30


def distribution(values):
    assert values and all(math.isfinite(v) for v in values)
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values),n=len(values))


def audit_window(r,*,name,batch,capacity,contexts,steps,repeats,modes,maximum):
    assert r['method']==name and r['batch']==batch and r['steps']==steps
    context=r['nominal_context']; repeat=r['repeat']; mode=r['mode']
    assert context in contexts and 0<=repeat<repeats and mode in modes
    start=context+(repeat*len(modes)+modes.index(mode))*steps
    assert r['logical_start']==start and r['logical_end']==start+steps
    elapsed=r['elapsed_seconds']; assert math.isfinite(elapsed) and elapsed>0
    assert math.isclose(r['ms_per_step'],1000*elapsed/steps,rel_tol=1e-10)
    assert math.isclose(r['tokens_per_second'],batch*steps/elapsed,rel_tol=1e-10)
    expected_events=0 if name.startswith('native_') else 8*max(0,(start+steps-capacity)//64)
    assert r['eviction_layer_events']==expected_events
    c=r['cache_bytes']; assert all(isinstance(x,int) and x>=0 for x in c.values())
    slots=start+steps if name=='native_dynamic' else capacity+64
    assert c['kv']==batch*32768*slots
    assert c['recurrent']==batch*48*MIB and c['conv']==batch*3*MIB//2
    assert c['positions']==(0 if name=='native_dynamic' else batch*256*slots)
    assert c['queries']==(batch*4*MIB if name.startswith('snapkv_pp') else 0)
    assert r['peak_reserved_bytes']>=r['peak_allocated_bytes']>=sum(c.values())
    if name=='native_preallocated':
        assert capacity==maximum and r['no_eviction_full_logit_max_error']==0
    if 'no_eviction_full_logit_max_error' in r:
        assert expected_events==0 and r['no_eviction_full_logit_max_error']==0
    return context,repeat,mode


def analyze(root):
    manifest=json.loads((root/'manifest.json').read_text()); settings=manifest['settings']; args=settings['args']
    for name,digest in settings['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Source changed: {name}'
    contexts=list(map(int,args['contexts'].split(','))); batches=list(map(int,args['batches'].split(',')))
    modes=args['modes'].split(','); steps=args['steps']; repeats=args['repeats']
    maximum=settings['maximum_logical_length']
    assert maximum==max(contexts)+steps*repeats*len(modes)
    inputs=json.loads((root/'input.json').read_text())
    assert len(inputs['prompt_ids'])+len(inputs['forced_ids'])==maximum
    assert hashlib.sha256(json.dumps([inputs['prompt_ids'],inputs['forced_ids']]).encode()).hexdigest()==inputs['input_sha256']
    records={}; runs=[]; files=[]
    expected={(c,r,m) for c in contexts for r in range(repeats) for m in modes}
    for batch in batches:
        records[batch]={}
        for spec in args['cells'].split(','):
            policy,_,cap=spec.partition(':')
            name=policy if policy.startswith('native_') else f'{policy}_C{cap}'
            capacity=maximum if name=='native_preallocated' else None if name=='native_dynamic' else int(cap)
            path=root/f'B{batch}'/f'{name}.json'; files.append(path)
            run=json.loads(path.read_text())
            assert run['status']=='complete' and run['method']==name and run['batch']==batch
            assert run['backing_capacity']==capacity
            assert run['reserved_slots_per_head']==(None if capacity is None else capacity+64)
            cell={}
            for row in run['records']:
                key=audit_window(row,name=name,batch=batch,capacity=capacity,contexts=contexts,
                    steps=steps,repeats=repeats,modes=modes,maximum=maximum)
                assert key not in cell; cell[key]=row
            assert set(cell)==expected
            assert run['overall_peak_allocated_bytes']>=max(r['peak_allocated_bytes'] for r in cell.values())
            if name=='native_dynamic':
                ref=root/f'B{batch}'/'native_logits.pt'
                assert hashlib.sha256(ref.read_bytes()).hexdigest()==run['reference_sha256']
            elif 'native_dynamic' in args['cells'].split(','):
                assert all(r.get('no_eviction_full_logit_max_error')==0 for r in cell.values() if r['eviction_layer_events']==0)
            records[batch][name]=cell
            runs.append({k:run[k] for k in ['method','batch','total_wall_seconds','overall_peak_allocated_bytes','gpu_before','gpu_after']})
    assert set(root.glob('B*/*.json'))==set(files),'Extra/unexpected cell files'
    summary=[]; ratios=[]
    for batch,cells in records.items():
        for name,cell in cells.items():
            for context in contexts:
                for mode in modes:
                    rows=[cell[(context,r,mode)] for r in range(repeats)]
                    summary.append(dict(batch=batch,method=name,nominal_context=context,mode=mode,
                        logical_windows=[[r['logical_start'],r['logical_end']] for r in rows],
                        tokens_per_second=distribution([r['tokens_per_second'] for r in rows]),
                        ms_per_step=distribution([r['ms_per_step'] for r in rows]),
                        peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/GIB,
                        peak_reserved_gib=max(r['peak_reserved_bytes'] for r in rows)/GIB,
                        kv_backing_mib_per_sequence=rows[-1]['cache_bytes']['kv']/batch/MIB,
                        recurrent_mib_per_sequence=rows[-1]['cache_bytes']['recurrent']/batch/MIB,
                        conv_mib_per_sequence=rows[-1]['cache_bytes']['conv']/batch/MIB))
                    for reference in ('native_dynamic','native_preallocated'):
                        if reference not in cells or reference==name: continue
                        rr=[cells[reference][(context,r,mode)] for r in range(repeats)]
                        assert all((a['logical_start'],a['logical_end'])==(b['logical_start'],b['logical_end']) for a,b in zip(rows,rr))
                        ratios.append(dict(batch=batch,method=name,reference=reference,nominal_context=context,mode=mode,
                            paired_throughput_ratio=distribution([a['tokens_per_second']/b['tokens_per_second'] for a,b in zip(rows,rr)])))
    h=hashlib.sha256()
    for path in sorted(files+[root/'input.json']):
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return dict(audit='passed',input_path=str(root),manifest_commit=manifest['git_commit'],
        completed_data_sha256=h.hexdigest(),input_token_sha256=inputs['input_sha256'],
        n_cells=len(runs),n_windows=sum(len(c) for cells in records.values() for c in cells.values()),
        total_run_wall_hours=sum(r['total_wall_seconds'] for r in runs)/3600,
        summaries=summary,paired_ratios=ratios,runs=runs,
        notes=['Model plus LM head on fixed teacher-forced inputs; excludes sampling, grading, EOS and serving scheduling.',
               'Matched logical windows at each batch. Three adjacent windows are not independent workloads; median/range only.',
               'Native preallocation reserves the maximum length at every point; shorter-context backing memory exceeds live KV.',
               'Decode-window peak and whole-run peak are separate; PyTorch allocated/reserved bytes exclude some device overhead.',
               'Native-preallocated full-vocabulary endpoint logits must exactly match DynamicCache.',
               'No-eviction preallocation also retains adapter position bookkeeping; its difference from DynamicCache is not allocator-only.',
               'Compressed-cache quality loss in the separate MATH study must accompany speed/memory results.'])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); report=analyze(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('Audit passed:',report['n_cells'],'cells,',report['n_windows'],'windows,',round(report['total_run_wall_hours'],3),'hours')
    for r in report['summaries']:
        if r['mode']=='throughput':
            print(r['method'],'B',r['batch'],'N',r['nominal_context'],'tok/s',round(r['tokens_per_second']['median'],1),'peakGiB',round(r['peak_allocated_gib'],3))


if __name__=='__main__': main()
