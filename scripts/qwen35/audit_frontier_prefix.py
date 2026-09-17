"""CPU-only paired prefix gate for an immutable completed frontier prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.qwen35.analyze_frontier import BASELINE_SHA,CELLS,data_hash,frozen_sources,check_settings,check_prefix


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path)
    ap.add_argument('--reference',default='results/qwen35/main32k_b2_v3',type=Path)
    ap.add_argument('--problems',type=int,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    current=json.loads((args.root/'manifest.json').read_text())['settings']
    reference=json.loads((args.reference/'manifest.json').read_text())['settings']
    check_settings(current,reference); frozen_sources(current)
    assert data_hash(args.reference)==BASELINE_SHA
    assert 0<args.problems<=len(current['problem_ids'])
    h=hashlib.sha256(); tokens=0; prefixes=0
    for pid in current['problem_ids'][:args.problems]:
        stem=hashlib.sha256(pid.encode()).hexdigest()[:20]
        for run in (0,1):
            native=json.loads((args.reference/'native'/f'{stem}_r{run}.json').read_text())
            for name,capacity in CELLS.items():
                relative=Path(name)/f'{stem}_r{run}.json'
                raw=(args.root/relative).read_bytes(); row=json.loads(raw)
                assert row['status']=='complete' and not row['grading_errors']
                assert row['capacity']==capacity and row['policy']=='random_pp' and row['recent']==64
                tokens+=check_prefix(row,native); prefixes+=1
                h.update(str(relative).encode()); h.update(raw)
    report=dict(status='passed',scope='Operational paired-prefix audit only; no hypothesis tests',
        n_problems=args.problems,n_matched_prefixes=prefixes,n_matched_tokens=tokens,
        native_data_sha256=BASELINE_SHA,selected_data_sha256=h.hexdigest(),
        source_and_generation_settings_match=True)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
