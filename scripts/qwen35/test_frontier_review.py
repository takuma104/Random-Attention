import hashlib
import pytest
from scripts.qwen35.analyze_frontier_review import apply_packet


def fixture(box='864'):
    row=dict(problem_id='p',gold='864 inches^2',final_boxed=box,final_correct=False,final_text='864 square inches')
    tid=hashlib.sha256(row['final_text'].encode()).hexdigest()[:20]
    packet=dict(case_id='case_x',problem_id='p',gold=row['gold'],boxed=box)
    ref=dict(old_final_correct=False,final_text_id=tid)
    return row,packet,ref,tid


def test_context_must_be_reviewed_and_raw_score_not_mutated():
    row,packet,ref,tid=fixture()
    decision=dict(category='contextual_units',new_final_correct=True,reviewed_final_text_variants=[])
    with pytest.raises(AssertionError): apply_packet(row,packet,decision,ref)
    decision['reviewed_final_text_variants']=[tid]
    result=apply_packet(row,packet,decision,ref)
    assert result['changed'] and row['review_final_correct'] and not row['final_correct']


def test_ambiguity_preserves_legacy_and_missing_box_cannot_be_rescued():
    row,packet,ref,_=fixture()
    decision=dict(category='ambiguity',new_final_correct=None)
    result=apply_packet(row,packet,decision,ref)
    assert result['unresolved'] and not result['changed'] and not row['review_final_correct']
    row,packet,ref,_=fixture(box=None)
    with pytest.raises(ValueError): apply_packet(row,packet,dict(category='incorrect',new_final_correct=True),ref)
