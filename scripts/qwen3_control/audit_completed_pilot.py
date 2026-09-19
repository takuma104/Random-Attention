"""Reproduce completed-pilot integrity and historical snapshot checks on CPU."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
from scripts.qwen3_control.analyze_math import ROOT,audit,digest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args(); checks={}
    for n in (10,25):
        old=json.loads((ROOT/f'docs/plans/qwen3-control-progress-{n}.json').read_text())
        _,_,new=audit(args.root,problems=n)
        assert old['data_sha256']==new['data_sha256']
        checks[str(n)]=new['data_sha256']
    a=json.loads((ROOT/'docs/plans/qwen3-control-8k-32k-prefix-early.json').read_text())
    b=json.loads((ROOT/'docs/plans/qwen3-control-8k-32k-prefix-audit.json').read_text()); assert a==b
    s,rows,report=audit(args.root)
    assert s['args']['phase']=='pilot' and report['n_answers']==300
    report.update(earlier_prefixes_unchanged=checks,extension_audit_unchanged=True,
        generation_hours=sum(r['elapsed_seconds'] for cell in rows.values() for r in cell.values())/3600,
        started_at_unix=min(r['started_at'] for cell in rows.values() for r in cell.values()),
        generation_finished_at_unix=max(r['started_at']+r['batch_wall_seconds'] for cell in rows.values() for r in cell.values()),
        raw_answer_bytes=sum(p.stat().st_size for p in args.root.glob('*/*.json')),
        source_sha256=digest(Path(__file__)),auditor_sha256=digest(ROOT/'scripts/qwen3_control/analyze_math.py'))
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
