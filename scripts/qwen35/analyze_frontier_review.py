"""Separate frontier score overlay with the original native adjudication frozen."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.qwen35.analyze_math import audit_and_load
from scripts.qwen35.analyze_frontier import (ROOT,BASELINE_SHA,CELLS,data_hash,load_reference,
    frozen_sources,check_settings,bootstrap_contrast,preservation)
from scripts.qwen35.analyze_grading_review import review_value


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(cell):
    for r in cell.values():
        r['review_final_correct']=r['final_correct']
        r['review_unresolved']=False


def apply_packet(row,packet,decision,ref):
    assert row['problem_id']==packet['problem_id'] and row['gold']==packet['gold']
    assert row['final_boxed']==packet['boxed'] and row['final_correct']==ref['old_final_correct']
    assert hashlib.sha256(row['final_text'].encode()).hexdigest()[:20]==ref['final_text_id']
    if decision['category'] in ('contextual_units','contextual_base_suffix'):
        assert ref['final_text_id'] in decision['reviewed_final_text_variants']
    row['review_final_correct']=review_value(row['final_correct'],decision['new_final_correct'],row['final_boxed'])
    row['review_unresolved']=decision['new_final_correct'] is None
    return dict(ref,case_id=packet['case_id'],new_final_correct=row['review_final_correct'],
        changed=row['review_final_correct']!=row['final_correct'],
        unresolved=row['review_unresolved'],category=decision['category'])


def analyze(root,review,decisions_path,reference_root):
    reference_manifest,native=load_reference(reference_root)
    manifest,records=audit_and_load(root); assert set(records)==set(CELLS)
    frozen_sources(manifest['settings']); check_settings(manifest['settings'],reference_manifest['settings'])
    before=data_hash(root)
    legacy=json.loads((ROOT/'docs/plans/qwen35-frontier32k-analysis.json').read_text())
    assert before==legacy['completed_data_sha256']
    meta=json.loads((review/'manifest.json').read_text())
    assert sha(root/'manifest.json')==meta['manifest_sha256']
    assert sha(review/'packets.json')==meta['packets_sha256']
    assert sha(review/'private_mapping.json')==meta['mapping_sha256']
    packets={p['case_id']:p for p in json.loads((review/'packets.json').read_text())}
    mapping=json.loads((review/'private_mapping.json').read_text())
    decisions=json.loads(decisions_path.read_text()); assert decisions['packets_sha256']==meta['packets_sha256']
    choices={d['case_id']:d for d in decisions['decisions']}
    assert len(choices)==len(decisions['decisions']) and set(choices)==set(packets)==set(mapping)
    data_path=Path(manifest['settings']['artifacts']['dataset_path'])
    assert sha(data_path)==manifest['settings']['artifacts']['dataset_sha256']
    questions={r['experiment_id']:r['problem'] for r in map(json.loads,data_path.read_text().splitlines())}
    for p in packets.values(): assert p['question']==questions[p['problem_id']]
    for cell in [native,*records.values()]: initialize(cell)
    # Existing native judgments are reused verbatim, never re-adjudicated here.
    frozen_path=ROOT/'docs/plans/qwen35-main32k-grading-sensitivity.json'
    frozen=json.loads(frozen_path.read_text()); assert frozen['completed_data_sha256']==BASELINE_SHA
    assert frozen['decisions_sha256']==sha(ROOT/'docs/plans/qwen35-grading-decisions-v1.json')
    native_sidecar=[]
    for entry in frozen['record_sidecar']:
        if Path(entry['path']).parent.name!='native': continue
        path=reference_root/entry['path']; assert sha(path)==entry['record_sha256']
        raw=json.loads(path.read_text()); row=native[(raw['problem_id'],raw['run'])]
        assert row['final_correct']==entry['old_final_correct']
        row['review_final_correct']=review_value(row['final_correct'],entry['new_final_correct'],row['final_boxed'])
        row['review_unresolved']=entry['unresolved']; native_sidecar.append(entry)
    assert sum(r['review_final_correct'] for r in native.values())/1000==frozen['cells']['native']['reviewed_final_accuracy']
    sidecar=[]; sampled=0; false_positives=0
    for cid,refs in mapping.items():
        for ref in refs:
            path=root/ref['path']; assert sha(path)==ref['record_sha256']
            raw=json.loads(path.read_text()); row=records[path.parent.name][(raw['problem_id'],raw['run'])]
            text_path=review/'texts'/f'{cid}_{ref["final_text_id"]}.txt'
            assert text_path.read_text()==raw['final_text']
            sidecar.append(apply_packet(row,packets[cid],choices[cid],ref))
            if 'accepted_audit_sample' in ref['selection_reasons']:
                sampled+=1; false_positives+=not row['review_final_correct']
    assert sampled==100 and len(sidecar)==meta['n_review_records']
    all_cells={'native':native,**records}; cells={}
    for name,cell in all_cells.items():
        rows=list(cell.values()); n=len(rows)
        for row in rows:
            row['review_upper_correct']=row['review_final_correct'] or row['review_unresolved']
        cells[name]=dict(n=n,legacy_final_accuracy=sum(r['final_correct'] for r in rows)/n,
            reviewed_final_accuracy=sum(r['review_final_correct'] for r in rows)/n,
            reviewed_eos_required_accuracy=sum(r['review_final_correct'] and r['termination']=='eos' for r in rows)/n,
            n_upgraded=sum(r['review_final_correct'] and not r['final_correct'] for r in rows),
            n_downgraded=sum(r['final_correct'] and not r['review_final_correct'] for r in rows),
            n_unresolved=sum(r['review_unresolved'] for r in rows),
            accuracy_if_all_unresolved_accepted=sum(r['review_upper_correct'] for r in rows)/n,
            n_eos_wrong_after_review=sum(r['termination']=='eos' and not r['review_final_correct'] for r in rows))
    contrasts={name:bootstrap_contrast(cell,native,'review_final_correct') for name,cell in records.items()}
    upper={name:bootstrap_contrast(cell,native,'review_upper_correct') for name,cell in records.items()}
    assert data_hash(root)==before and data_hash(reference_root)==BASELINE_SHA
    return dict(analysis='Separate curated grading sensitivity, not replacement primary scores',audit='passed',
        completed_data_sha256=before,reference_data_sha256=BASELINE_SHA,
        frozen_native_review_sha256=sha(frozen_path),decisions_sha256=sha(decisions_path),review_metadata=meta,
        n_changed_new=sum(r['changed'] for r in sidecar),
        new_changes_by_category=dict(Counter(r['category'] for r in sidecar if r['changed'])),
        accepted_audit_sample_n=sampled,observed_sample_false_positives=false_positives,cells=cells,
        contrasts=contrasts,same_protocol_criteria_applied_as_sensitivity={n:preservation(s) for n,s in contrasts.items()},
        contrasts_if_all_unresolved_accepted=upper,record_sidecar=sidecar,frozen_native_sidecar=native_sidecar,
        limitations=['AI-assisted, not independent human adjudication or general validated grader.',
            'Condition/run mapping was not consulted until decisions were committed; prior main review and frontier aggregate results were known.',
            'New frontier formatting judgments are sensitivity only; original native review judgments unchanged.',
            'Gold-exact matches do not re-prove dataset correctness. Only100 accepted new answers sampled for false positives.',
            'Missing/empty boxes not rescued. Ambiguities retain legacy; upper scenario accepts ambiguities on both sides.',
            'Same reused MATH500/native dataset, not independent replication; old C2048 primary failure unchanged.'])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path)
    ap.add_argument('--review',required=True,type=Path); ap.add_argument('--decisions',required=True,type=Path)
    ap.add_argument('--reference',type=Path,default=Path('results/qwen35/main32k_b2_v3'))
    ap.add_argument('--out',required=True,type=Path); args=ap.parse_args()
    report=analyze(args.root,args.review,args.decisions,args.reference)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['n_changed_new','new_changes_by_category','observed_sample_false_positives','cells','same_protocol_criteria_applied_as_sensitivity']},indent=2))


if __name__=='__main__': main()
