"""Tests for overlay safeguards; never import the GPU inference runner."""
import pytest
from scripts.qwen35.analyze_grading_review import review_value


def test_unresolved_keeps_original():
    assert review_value(False,None,'extra roots') is False
    assert review_value(True,None,'gold') is True


def test_no_missing_answer_rescue():
    for boxed in (None,''):
        with pytest.raises(ValueError):
            review_value(False,True,boxed)
    assert review_value(False,False,None) is False


def test_exact_boolean_decisions():
    for invalid in ('true',1,0,[]):
        with pytest.raises(ValueError):
            review_value(False,invalid,'east')
    assert review_value(False,True,'east') is True
    assert review_value(True,False,'wrong') is False
