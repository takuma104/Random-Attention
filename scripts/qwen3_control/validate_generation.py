"""Non-benchmark arithmetic smoke: native EOS, paired prefix and legacy grading.

The 256-token numeric-validation samples all capped and did not exercise EOS.
This separate smoke uses a fixed 4096 cap, not any MATH500 question.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen3_control.runtime import ROOT,load_model,prompt_tokens,generate_batch,source_hashes
from kvcompress.qwen3_control.cache import install_eviction_hooks
from scripts.qwen35.run_math import atomic_json
import argparse
import hashlib
import json
import subprocess
import torch


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True,type=Path); args=ap.parse_args()
    assert not args.out.exists()
    model,tokenizer,info=load_model(); handles=install_eviction_hooks(model)
    ids=prompt_tokens(tokenizer,'Compute 17 + 25.').expand(2,-1)
    samples={}; summary={}
    short_path=ROOT/'work/qwen3_control/qwen3-control-validation-v2-samples.json'
    short=json.loads(short_path.read_text())
    for policy,capacity in [('native',0),('random_pp',1024)]:
        torch.cuda.empty_cache()
        result=generate_batch(model,ids,policy=policy,capacity=capacity,recent=64,max_new_tokens=4096,
            generation_seeds=[101,202],eviction_seeds=[303,404])
        samples[policy]=[]; summary[policy]=[]
        for row,(tokens,metrics) in enumerate(result):
            previous=short[policy][row][0]
            assert tokens[:len(previous)]==previous, '256-to-4096 prefix mismatch'
            assert not any(t in metrics['eos_ids'] for t in tokens[:-1])
            assert (tokens[-1] in metrics['eos_ids'])==(metrics['termination']=='eos')
            assert metrics['logical_cached_tokens']==ids.shape[1]+len(tokens)-1
            expected=0 if policy=='native' else 36*max(0,(metrics['logical_cached_tokens']-capacity)//64)
            assert metrics['eviction_layer_events']==expected
            text=tokenizer.decode(tokens,skip_special_tokens=False)
            grade=json.loads(subprocess.run([sys.executable,'-m','kvcompress.qwen35.grade'],
                input=json.dumps(dict(completion=text,gold='42',termination=metrics['termination'])),
                text=True,capture_output=True,timeout=30,check=True,cwd=ROOT).stdout)
            assert not grade['grading_errors']
            if policy!='native':
                base=samples['native'][row]['tokens']; n=min(len(base),capacity+64-ids.shape[1]+1)
                assert tokens[:n]==base[:n]
            samples[policy].append(dict(tokens=tokens,completion=text,metrics=metrics,grade=grade))
            summary[policy].append(dict(tokens=len(tokens),termination=metrics['termination'],
                final_correct=grade['final_correct'],eviction_layer_events=expected,
                token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))
    for h in handles: h.remove()
    raw=ROOT/'work/qwen3_control'/f'{args.out.stem}-samples.json'
    assert not raw.exists(); atomic_json(raw,samples)
    # Preserve diagnostic output even if the longer fixed-cap native smoke caps.
    passed=all(r['termination']=='eos' for r in summary['native'])
    hashes=source_hashes(); hashes[str(Path(__file__).resolve().relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(args.out,dict(status='passed' if passed else 'native_eos_not_observed',model_revision=info['model_revision'],
        source_hashes=hashes,raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),summary=summary,
        scope='Fixed arithmetic implementation smoke, NOT MATH500 accuracy evidence',prefix_matches_native=True,
        extension_256_to_4096_prefix_exact=True,extension_reference_sha256=hashlib.sha256(short_path.read_bytes()).hexdigest()))
    print(json.dumps(summary,indent=2),flush=True)
    assert passed, 'Investigate missing native EOS; do not silently increase the cap or change sampling'


if __name__=='__main__': main()
