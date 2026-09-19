"""CPU-only audit that fixed smoke trajectories extend unchanged in the pilot."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
from scripts.qwen3_control.analyze_math import ROOT,audit,digest


def check_prefix(short,long):
    for field in ('problem_id','run','prompt_token_ids','generation_seed','eviction_seed',
                  'policy','capacity','recent','sampling','eos_ids'):
        assert short[field]==long[field],field
    n=len(short['token_ids'])
    assert long['token_ids'][:n]==short['token_ids'], 'Output-cap extension prefix drift'
    if short['termination']=='eos':
        assert long['termination']=='eos' and len(long['token_ids'])==n
    return n


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('short',type=Path); ap.add_argument('long',type=Path)
    ap.add_argument('--out',required=True,type=Path); args=ap.parse_args()
    s,a,ar=audit(args.short); t,b,br=audit(args.long,problems=2)
    assert s['args']['phase']=='smoke' and t['args']['phase']=='pilot'
    for field in s:
        if field not in ('args','problem_ids'): assert s[field]==t[field],field
    for field in s['args']:
        if field not in ('phase','out','limit','max_new_tokens'): assert s['args'][field]==t['args'][field],field
    total=count=eos=0
    for cell,values in a.items():
        for key,row in values.items():
            total+=check_prefix(row,b[cell][key]); count+=1; eos+=row['termination']=='eos'
    report=dict(status='passed',short_manifest_sha256=ar['manifest_sha256'],long_manifest_sha256=br['manifest_sha256'],
        short_data_sha256=ar['data_sha256'],long_two_problem_data_sha256=br['data_sha256'],
        exact_prefixes=count,exact_prefix_tokens=total,already_eos_unchanged=eos,
        source_sha256=digest(Path(__file__)),auditor_sha256=digest(ROOT/'scripts/qwen3_control/analyze_math.py'))
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
