"""Prepare condition-masked CPU-only post-hoc review packets; never alter results.

All EOS-incorrect answers, all incorrect nonempty final boxes, all cap-correct
answers, and 100 hash-selected accepted answers are reviewed. Identical problem/
boxed-answer pairs share a packet, but every distinct final-text variant remains
available for contextual review. Mappings to methods remain in a separate file.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.qwen35.analyze_math import audit_and_load


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path('work/qwen35/math500.jsonl'))
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new review directory, preserving earlier packets/decisions')
    manifest,records=audit_and_load(args.root)
    assert len(manifest['settings']['problem_ids'])==500
    data={r['experiment_id']:r for r in map(json.loads,args.data.read_text().splitlines())}
    answers=[]
    for cell,rows in records.items():
        for (pid,run),row in rows.items():
            name=f'{cell}/{digest(pid.encode())[:20]}_r{run}.json'
            raw=(args.root/name).read_bytes()
            answers.append((name,row,digest(raw)))
    accepted=[r for r in answers if r[1]['final_correct']]
    sample={r[0] for r in sorted(accepted,key=lambda r:digest(('20260911|'+r[0]).encode()))[:100]}
    groups=defaultdict(list)
    for name,row,sha in answers:
        reasons=[]
        if not row['final_correct'] and row['termination']=='eos': reasons.append('eos_incorrect')
        if not row['final_correct'] and row['final_boxed'] is not None: reasons.append('incorrect_box')
        if row['final_correct'] and row['termination']=='length': reasons.append('cap_correct')
        if name in sample: reasons.append('accepted_audit_sample')
        if reasons:
            groups[(row['problem_id'],row['final_boxed'])].append((name,row,sha,reasons))
    args.out.mkdir(parents=True)
    (args.out/'texts').mkdir()
    packets=[]; mapping={}
    for (pid,boxed),rows in groups.items():
        case_id='case_'+digest(json.dumps([pid,boxed],ensure_ascii=False).encode())[:16]
        variants={}
        for _,row,_,_ in rows:
            text=row['final_text']
            text_id=digest(text.encode())[:20]
            variants[text_id]=text
        for text_id,text in variants.items():
            (args.out/'texts'/f'{case_id}_{text_id}.txt').write_text(text)
        packets.append(dict(case_id=case_id,problem_id=pid,question=data[pid]['problem'],gold=data[pid]['answer'],
                            boxed=boxed,final_text_variants=sorted(variants),
                            selection_reasons=sorted({reason for *_,reasons in rows for reason in reasons})))
        mapping[case_id]=[dict(path=name,record_sha256=sha,old_final_correct=row['final_correct'],
                              final_text_id=digest(row['final_text'].encode())[:20],selection_reasons=reasons)
                          for name,row,sha,reasons in rows]
    packets.sort(key=lambda r:r['case_id'])
    (args.out/'packets.json').write_text(json.dumps(packets,indent=2,ensure_ascii=False)+'\n')
    (args.out/'private_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    metadata=dict(input_path=str(args.root),manifest_sha256=digest((args.root/'manifest.json').read_bytes()),
        packets_sha256=digest((args.out/'packets.json').read_bytes()),
        mapping_sha256=digest((args.out/'private_mapping.json').read_bytes()),
        n_answers=len(answers),n_review_records=sum(len(x) for x in mapping.values()),n_packets=len(packets),
        accepted_sample_size=len(sample),accepted_sample_seed='20260911',
        masking='Packets omit condition/run; separate mapping must not be consulted until decisions are recorded.',
        prior_exposure='First-100 false negatives were already reviewed with conditions visible; not fully blinded.')
    (args.out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
