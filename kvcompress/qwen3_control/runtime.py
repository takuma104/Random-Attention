"""Pinned Qwen3-4B control using the unchanged shared sampler/cache primitives."""
# Shared primitives import Qwen3.5 modeling; preserve its safe import order even
# though this control has no DeltaNet layers or FLA forward calls.
from kvcompress.qwen35.runtime import ROOT,prompt_tokens,resolve_eos_ids
import hashlib
import json
from dataclasses import asdict
import torch
from transformers import AutoTokenizer,Qwen3ForCausalLM
from kvcompress.qwen35.generate import SamplingConfig,stable_seed
from kvcompress.qwen35.batch import generate_batch as _generate_batch
from scripts.qwen35.run_math import source_hashes as shared_hashes

SAMPLING=SamplingConfig(temperature=.6,top_p=.95,top_k=20,presence_penalty=0.)


def artifacts():
    return json.loads((ROOT/'work/qwen3_control/artifacts.json').read_text())


def source_hashes():
    result=shared_hashes()
    for path in sorted((ROOT/'kvcompress/qwen3_control').glob('*.py')):
        result[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def load_model():
    info=artifacts(); path=info['model_path']
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model,loading=Qwen3ForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa',output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys')),loading
    assert model.config.model_type=='qwen3' and model.config.num_hidden_layers==36
    assert model.config.layer_types==['full_attention']*36
    assert model.config.head_dim==128 and model.model.rotary_emb.rope_type=='default'
    cfg=model.generation_config
    assert (cfg.temperature,cfg.top_p,cfg.top_k,getattr(cfg,'presence_penalty',0.))==(.6,.95,20,0.)
    cfg.eos_token_id=resolve_eos_ids(cfg.eos_token_id,tokenizer.eos_token_id)
    assert cfg.eos_token_id==[151645,151643]
    model.eval()
    print('Qwen3 control:',info['model_revision'],'parameter_bytes',sum(p.numel()*p.element_size() for p in model.parameters()),
        'EOS',cfg.eos_token_id,'sampling',asdict(SAMPLING),flush=True)
    return model,tokenizer,info


def generate_batch(model,input_ids,*,sampling=SAMPLING,**kwargs):
    assert model.config.model_type=='qwen3'
    return _generate_batch(model,input_ids,sampling=sampling,**kwargs)
