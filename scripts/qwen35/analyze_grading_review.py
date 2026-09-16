"""Apply frozen, condition-masked post-hoc decisions to in-memory score overlays.

Original answer files and preregistered scores remain byte-for-byte unchanged.
This is a curated sensitivity analysis, NOT a replacement general-purpose grader.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.qwen35.analyze_math import audit_and_load,paired_stats
from scripts.qwen35.analyze_main import primary_decisions


def sha(data):
    return hashlib.sha256(data).hexdigest()


def review_value(old,decision,boxed):
    if decision is None:
        return old
    if type(decision) is not bool:
        raise ValueError('Decision must be bool or null')
    if decision and not boxed:
        raise ValueError('Cannot rescue an absent/empty final box')
    return decision


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--review',type=Path,required=True)
    ap.add_argument('--decisions',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    manifest,records=audit_and_load(args.root)
    meta=json.loads((args.review/'manifest.json').read_text())
    assert sha((args.root/'manifest.json').read_bytes())==meta['manifest_sha256']
    assert sha((args.review/'packets.json').read_bytes())==meta['packets_sha256']
    assert sha((args.review/'private_mapping.json').read_bytes())==meta['mapping_sha256']
    packets={p['case_id']:p for p in json.loads((args.review/'packets.json').read_text())}
    mapping=json.loads((args.review/'private_mapping.json').read_text())
    decisions=json.loads(args.decisions.read_text())
    assert decisions['packets_sha256']==meta['packets_sha256']
    choices={d['case_id']:d for d in decisions['decisions']}
    assert len(choices)==len(decisions['decisions']) and set(choices)==set(mapping)==set(packets)
    for cell in records.values():
        for row in cell.values():
            row['review_final_correct']=row['final_correct']
            row['review_final_correct_terminated']=row['final_correct_terminated']
            row['review_unresolved']=False
    sidecar=[]; sampled=0; sample_false_positive=0
    for cid,refs in mapping.items():
        decision=choices[cid]
        for ref in refs:
            path=args.root/ref['path']
            assert sha(path.read_bytes())==ref['record_sha256']
            raw=json.loads(path.read_text())
            assert raw['final_correct']==ref['old_final_correct']
            assert raw['problem_id']==packets[cid]['problem_id'] and raw['final_boxed']==packets[cid]['boxed']
            assert sha(raw['final_text'].encode())[:20]==ref['final_text_id']
            if decision['category'] in ('contextual_units','contextual_base_suffix','region_vs_volume_wording'):
                assert ref['final_text_id'] in decision['reviewed_final_text_variants']
            row=records[path.parent.name][(raw['problem_id'],raw['run'])]
            new=review_value(row['final_correct'],decision['new_final_correct'],row['final_boxed'])
            row['review_final_correct']=new
            row['review_final_correct_terminated']=new and row['termination']=='eos'
            row['review_unresolved']=decision['new_final_correct'] is None
            sidecar.append(dict(ref,case_id=cid,new_final_correct=new,
                                changed=new!=row['final_correct'],unresolved=row['review_unresolved'],
                                category=decision['category']))
            if 'accepted_audit_sample' in ref['selection_reasons']:
                sampled+=1
                sample_false_positive+=not new
    assert sampled==100
    cells={}
    for name,cell in records.items():
        rows=list(cell.values()); n=len(rows)
        cells[name]=dict(n=n,legacy_final_accuracy=sum(r['final_correct'] for r in rows)/n,
            reviewed_final_accuracy=sum(r['review_final_correct'] for r in rows)/n,
            reviewed_eos_required_accuracy=sum(r['review_final_correct_terminated'] for r in rows)/n,
            n_upgraded=sum(r['review_final_correct'] and not r['final_correct'] for r in rows),
            n_downgraded=sum(r['final_correct'] and not r['review_final_correct'] for r in rows),
            n_unresolved=sum(r['review_unresolved'] for r in rows),
            reviewed_accuracy_if_all_unresolved_accepted=sum(r['review_final_correct'] or r['review_unresolved'] for r in rows)/n,
            n_eos_incorrect_after_review=sum(r['termination']=='eos' and not r['review_final_correct'] for r in rows))
    pairs=[(name,'native') for name in records if name!='native']
    pairs += [('random_pp_C1024','recency_pp_C1024'),('random_pp_C1024','snapkv_pp_C1024')]
    contrasts={f'{a} - {b}':{'final_correct':paired_stats(records[a],records[b],'review_final_correct')}
               for a,b in pairs}
    aggregate=hashlib.sha256()
    for path in sorted(args.root.glob('*/*.json')):
        aggregate.update(str(path.relative_to(args.root)).encode()); aggregate.update(path.read_bytes())
    report=dict(analysis='POST-HOC curated grading sensitivity, not preregistered primary scores',
        input_path=str(args.root),original_manifest_sha256=meta['manifest_sha256'],
        completed_data_sha256=aggregate.hexdigest(),decisions_sha256=sha(args.decisions.read_bytes()),
        review_metadata=meta,n_changed=sum(r['changed'] for r in sidecar),
        changes_by_category=dict(Counter(r['category'] for r in sidecar if r['changed'])),
        unresolved_records=sum(r['unresolved'] for r in sidecar),accepted_audit_sample_n=sampled,
        observed_sample_false_positives=sample_false_positive,cells=cells,paired_contrasts=contrasts,
        same_protocol_criteria_applied_posthoc=primary_decisions(contrasts),record_sidecar=sidecar,
        limitations=['AI-assisted review, not an independent human adjudication or a validated universal grader.',
            'Exact matches assessed against supplied gold; dataset correctness itself not comprehensively re-proven.',
            'Condition/run identities masked in packets; previously viewed first-100 cases were already exposed.',
            'New interval/unit/ordinal equivalents encountered at full review are post-hoc additions, not preregistered rules.',
            'Interpretation ambiguities retain legacy scores; upper sensitivity accepts them all without declaring them correct.',
            'No missing boxes rescued; cap and EOS definitions unchanged; only100 accepted answers sampled for false positives.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['n_changed','changes_by_category','unresolved_records','observed_sample_false_positives','cells','same_protocol_criteria_applied_posthoc']},indent=2))


if __name__=='__main__':
    main()
