"""Post-diagnostic BF16 correctness gate against a same-input FP32 oracle.

Check EACH leading-axis group independently (row/head before projection, row
for projected output). Relative L2 and Linf both <=1%. This avoids elementwise
relative errors at cancellation-induced zeros without pooling groups.
Native/preeviction remain exact.
"""
import torch

LIMIT=.01


def check_bf16_reference(actual,reference):
    assert actual.shape==reference.shape and actual.ndim>=2
    a=actual.float().flatten(1); b=reference.float().flatten(1)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta=a-b
    l2=delta.norm(dim=1)/b.norm(dim=1).clamp_min(1e-12)
    linf=delta.abs().amax(dim=1)/b.abs().amax(dim=1).clamp_min(1e-12)
    result=dict(max_abs=delta.abs().max().item(),mean_abs=delta.abs().mean().item(),
        max_group_relative_l2=l2.max().item(),max_group_relative_linf=linf.max().item())
    assert (l2<=LIMIT).all() and (linf<=LIMIT).all(),result
    return result
