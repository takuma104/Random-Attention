"""Pinned experiment runtime helpers (import this BEFORE Qwen3.5 modeling).

FLA must be imported first: Transformers 5.16.1's optional dependency resolver
can otherwise silently select its torch-only DeltaNet fallback during a circular
import. Kernel provenance is audited explicitly rather than assuming installation
implies activation.
"""
import fla  # noqa: F401 -- intentionally initialize before Transformers modeling
from fla.ops.gated_delta_rule import (  # noqa: F401 -- initialize nested package too
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
import inspect
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5

ROOT = Path(__file__).resolve().parents[2]


def kernel_provenance():
    report = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
        obj = getattr(modeling_qwen3_5, name)
        func = inspect.getclosurevars(obj.forward).nonlocals["func"] if isinstance(obj, torch.nn.Module) else obj
        implementation = inspect.getclosurevars(func).nonlocals.get("implementation", func)
        report[name] = implementation.__module__ + "." + implementation.__name__
    return report


def load_model():
    artifacts = json.loads((ROOT / "work/qwen35/artifacts.json").read_text())
    provenance = kernel_provenance()
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        if not provenance[name].startswith("fla."):
            raise RuntimeError(f"Refusing slow DeltaNet fallback: {provenance}")
    print("Kernels:", provenance, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(artifacts["model_path"], local_files_only=True)
    model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
        artifacts["model_path"], dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True, output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("unexpected_keys"):
        raise RuntimeError(f"Checkpoint loading discrepancies: {loading}")
    model.eval()
    print("Model loaded; parameter bytes:", sum(p.numel()*p.element_size() for p in model.parameters()), flush=True)
    return model, tokenizer, artifacts


def prompt_tokens(tokenizer, question):
    messages = [{"role": "user", "content": question + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=True, return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
