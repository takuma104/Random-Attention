"""Distinguish BF16 batch-shape roundoff from a batching correctness defect.

Compares native batch1/batch2 on fixed tokens in BF16+FLA, then FP32 with the
upstream pure-torch DeltaNet reference. This diagnostic is NOT an accuracy run.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen35.runtime import ROOT,load_model,prompt_tokens
import inspect
import json
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling


@torch.inference_mode()
def collect(model,ids,continuation,batch):
    cache=DynamicCache(config=model.config)
    current=ids.expand(batch,-1)
    out=[]
    for step in range(8):
        logits=model(current,past_key_values=cache,use_cache=True,logits_to_keep=1).logits
        out.append(logits[0,0].float().cpu())
        if batch==2:
            torch.testing.assert_close(logits[0],logits[1],rtol=0,atol=0)
        current=continuation[:,step:step+1].expand(batch,-1)
    return torch.stack(out)


def comparison(a,b):
    logp=a.log_softmax(-1); logq=b.log_softmax(-1)
    return dict(max_abs=(a-b).abs().max().item(),mean_abs=(a-b).abs().mean().item(),
                max_kl=(logp.exp()*(logp-logq)).sum(-1).max().item(),
                argmax_agreement=(a.argmax(-1)==b.argmax(-1)).float().mean().item())


@torch.inference_mode()
def main():
    model,tokenizer,manifest=load_model()
    torch.set_float32_matmul_precision('highest')
    row=next(json.loads(s) for s in (ROOT/'work/qwen35/math500.jsonl').read_text().splitlines()
             if json.loads(s)['experiment_id']==manifest['pilot_ids'][0])
    ids=prompt_tokens(tokenizer,row['problem'])
    generated=next(json.loads(p.read_text()) for p in (ROOT/'results/qwen35/pilot_v2/native').glob('*.json')
                   if json.loads(p.read_text())['problem_id']==row['experiment_id'])
    continuation=torch.tensor([generated['token_ids'][:8]],device='cuda')
    bf1=collect(model,ids,continuation,1)
    bf2=collect(model,ids,continuation,2)
    report={'bf16_b1_vs_b2':comparison(bf1,bf2)}
    print('BF16',report,flush=True)
    for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
        obj=getattr(modeling,name)
        wrapper=inspect.getclosurevars(obj.forward).nonlocals['func']
        reference=inspect.getclosurevars(wrapper).nonlocals['torch_function']
        setattr(modeling,name,reference)
    model.float()
    torch.cuda.empty_cache()
    fp1=collect(model,ids,continuation,1)
    fp2=collect(model,ids,continuation,2)
    report.update(fp32_b1_vs_b2=comparison(fp1,fp2),bf16_b1_vs_fp32=comparison(bf1,fp1),
                  bf16_b2_vs_fp32=comparison(bf2,fp2),steps=8,
                  fp32_reference='upstream torch-only DeltaNet; float32 matmul precision highest')
    (ROOT/'work/qwen35/batch_numerics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    assert report['fp32_b1_vs_b2']['max_abs']<.005
    assert report['bf16_b1_vs_b2']['max_kl']<.005


if __name__=='__main__':
    main()
