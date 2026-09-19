import pytest
import torch
from scripts.qwen3_control.numerics import check_bf16_reference


def test_bf16_rounding_and_identical_zeros():
    x=torch.linspace(-200,200,128).reshape(2,1,64)
    check_bf16_reference(x.bfloat16(),x)
    check_bf16_reference(torch.zeros_like(x),torch.zeros_like(x))


def test_small_bad_row_cannot_hide_behind_large_row():
    reference=torch.tensor([[1e6,0.],[1.,0.]])
    actual=reference.clone(); actual[1,0]=1.02
    with pytest.raises(AssertionError): check_bf16_reference(actual,reference)


def test_nonfinite_rejected():
    with pytest.raises(AssertionError): check_bf16_reference(torch.tensor([[float('nan')]]),torch.ones(1,1))
