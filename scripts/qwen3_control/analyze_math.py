"""CPU-only Qwen3 control integrity/progress audit; optional final pilot contrasts."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CELLS={'native':0,'random_pp_C1024':1024,'random_pp_C2048':2048}
SAMPLING=dict(temperature=.6,top_p=.95,top_k=20,presence_penalty=0.,repetition_penalty=1.,min_p=0.)


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def seed(pid,run,stream):
    # Historical namespace intentionally retained, same as frozen stable_seed.
    return int.from_bytes(hashlib.sha256(f'qwen35-experiment-v1:{pid}:{run}:{stream}'.encode()).digest()[:8],'little')%(2**63-1)


def check_record(r,cap,max_tokens):
    p=r['prompt_tokens']; n=r['generated_tokens']; logical=p+n-1
    assert r['status']=='complete' and not r['grading_errors']
    assert r['batch_size']==2 and r['batch_row']==r['run'] and r['run'] in (0,1)
    assert r['sampling']==SAMPLING and r['recent']==64 and r['capacity']==cap
    assert r['policy']==('native' if cap==0 else 'random_pp')
    assert p==len(r['prompt_token_ids']) and 0<n==len(r['token_ids'])<=max_tokens
    assert r['eos_ids']==[151645,151643] and r['logical_cached_tokens']==logical
    assert not any(t in r['eos_ids'] for t in r['token_ids'][:-1])
    assert r['termination'] in ('eos','length')
    assert (r['token_ids'][-1] in r['eos_ids'])==(r['termination']=='eos')
    if r['termination']=='length': assert n==max_tokens
    assert r['final_correct_terminated']==bool(r['final_correct'] and r['termination']=='eos')
    for stream in ('generation','eviction'): assert r[f'{stream}_seed']==seed(r['problem_id'],r['run'],stream)
    ev=0 if not cap else max(0,(logical-cap)//64)
    assert r['eviction_layer_events']==36*ev
    if cap:
        assert p<cap-64 and r['final_physical_lengths']==[logical-64*ev]*36
    else: assert r['final_physical_lengths']==[]
    expected=dict(kv=147456*(cap+64 if cap else logical),recurrent=0,conv=0,
                  positions=2304*(cap+64) if cap else 0,queries=0)
    assert r['cache_bytes']==expected
    assert r['batch_cache_bytes_at_finish']=={k:2*v for k,v in expected.items()}
    for key in ('elapsed_seconds','batch_wall_seconds','request_latency_seconds','ttft_seconds'):
        assert math.isfinite(r[key]) and r[key]>0
    assert r['ttft_seconds']<=r['request_latency_seconds']<=r['batch_wall_seconds']
    assert r['elapsed_seconds']*2==r['batch_wall_seconds']


def audit(root,problems=None):
    manifest=json.loads((root/'manifest.json').read_text()); s=manifest['settings']
    assert s['batch_size']==2 and s['full_attention_layers']==36 and s['sampling']==SAMPLING
    assert s['args']['runs']==2 and s['args']['cells']=='native,random_pp:1024,random_pp:2048'
    assert s['artifacts']['model_id']=='Qwen/Qwen3-4B'
    assert s['artifacts']['config']['model_type']=='qwen3'
    assert digest(s['artifacts']['dataset_path'])==s['artifacts']['dataset_sha256']
    assert digest(ROOT/'docs/plans/qwen3-control-pilot-protocol.md')==s['protocol_sha256']
    for group in ('source_hashes','installed_source_hashes','validation_hashes'):
        for path,sha in s[group].items(): assert digest(ROOT/path)==sha,path
    all_ids=s['problem_ids']; chosen=all_ids if problems is None else all_ids[:problems]
    assert chosen and len(set(all_ids))==len(all_ids)==s['args']['limit']
    if problems is not None: assert len(chosen)==problems
    assert all_ids==s['artifacts']['pilot_ids'][:len(all_ids)]
    expected_limit,expected_cap=(2,8192) if s['args']['phase']=='smoke' else (50,32768)
    assert s['args']['phase'] in ('smoke','pilot') and len(all_ids)==expected_limit
    assert s['args']['max_new_tokens']==expected_cap
    dataset={r['experiment_id']:r for r in map(json.loads,Path(s['artifacts']['dataset_path']).read_text().splitlines())}
    rows={}; data_hash=hashlib.sha256(); expected_paths=set()
    for cell,cap in CELLS.items():
        rows[cell]={}
        for pid in chosen:
            key=hashlib.sha256(pid.encode()).hexdigest()[:20]
            assert not (root/cell/f'{key}_g0.batch-ungraded').exists()
            pair=[]
            for run in (0,1):
                relative=Path(cell)/f'{key}_r{run}.json'; expected_paths.add(relative)
                payload=(root/relative).read_bytes(); r=json.loads(payload)
                data_hash.update(str(relative).encode()); data_hash.update(payload)
                assert r['problem_id']==pid and r['run']==run
                assert r['model_revision']==s['artifacts']['model_revision']
                assert r['gold']==dataset[pid]['answer'] and r['source_index']==dataset[pid]['source_index']
                assert r['batch_id']==f'{cell}/{key}_g0'
                check_record(r,cap,expected_cap)
                rows[cell][pid,run]=r; pair.append(r)
            for field in ('batch_wall_seconds','peak_allocated_bytes','peak_reserved_bytes','started_at','prompt_token_ids'):
                assert pair[0][field]==pair[1][field],field
    if problems is None:
        assert set(p.relative_to(root) for p in root.glob('*/*.json'))==expected_paths
        assert not list(root.glob('*/*.batch-ungraded'))
    prefix_count=prefix_tokens=0
    for cell,cap in CELLS.items():
        if not cap: continue
        for key,r in rows[cell].items():
            native=rows['native'][key]
            assert r['prompt_token_ids']==native['prompt_token_ids']
            n=min(native['generated_tokens'],cap+64-r['prompt_tokens']+1)
            assert r['token_ids'][:n]==native['token_ids'][:n],(cell,key,'prefix')
            prefix_count+=1; prefix_tokens+=n
    return s,rows,dict(status='passed',manifest_sha256=digest(root/'manifest.json'),data_sha256=data_hash.hexdigest(),
        n_problems=len(chosen),n_answers=sum(map(len,rows.values())),pre_eviction_prefixes=prefix_count,
        pre_eviction_prefix_tokens=prefix_tokens,frozen_sources_match=True,
        scope='Operational audit; no hypothesis tests or sample-size adaptation')


def describe(rows):
    result={}
    for cell,values in rows.items():
        rs=list(values.values()); n=len(rs); seconds=sum(r['elapsed_seconds'] for r in rs)
        groups=defaultdict(list)
        for r in rs: groups[r['batch_id']].append(r)
        total=sum(r['generated_tokens'] for r in rs)
        result[cell]=dict(n=n,final_accuracy=sum(r['final_correct'] for r in rs)/n,
            paper_accuracy=sum(r['paper_correct'] for r in rs)/n,
            eos_required_accuracy=sum(r['final_correct_terminated'] for r in rs)/n,
            cap_rate=sum(r['termination']=='length' for r in rs)/n,
            eos_wrong=sum(r['termination']=='eos' and not r['final_correct'] for r in rs),
            mean_tokens=total/n,eviction_exposure=sum(r['eviction_layer_events']>0 for r in rs)/n,
            generation_hours=seconds/3600,useful_tok_s=total/seconds,
            useful_row_fraction=total/sum(2*max(r['generated_tokens'] for r in g) for g in groups.values()),
            peak_shared_allocated_gib=max(r['peak_allocated_bytes'] for r in rs)/2**30)
    return result


def contrasts(rows,ids):
    import numpy as np
    # Same resampled problem clusters for both contrasts, keeping both runs.
    indices=np.random.default_rng(20260919).integers(0,len(ids),(10000,len(ids)))
    result={}
    for cell in ('random_pp_C1024','random_pp_C2048'):
        delta=np.array([sum(float(rows[cell][pid,r]['final_correct'])-float(rows['native'][pid,r]['final_correct'])
                            for r in (0,1))/2 for pid in ids])
        result[cell]=dict(difference=float(delta.mean()),ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
            problem_wins=int((delta>0).sum()),problem_losses=int((delta<0).sum()),problem_ties=int((delta==0).sum()))
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--problems',type=int)
    ap.add_argument('--final',action='store_true'); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); s,rows,report=audit(args.root,args.problems)
    report['cells']=describe(rows); report['analysis_source_sha256']=digest(Path(__file__))
    if args.final:
        assert args.problems is None and s['args']['phase']=='pilot'
        report.update(scope='Exploratory pilot; descriptive clustered intervals, not confirmatory preservation tests',
            contrasts=contrasts(rows,s['problem_ids']),bootstrap_replicates=10000,bootstrap_seed=20260919,
            independent_problem_clusters=50)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
