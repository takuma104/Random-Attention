"""Verify baseline bytes, frozen generation environment and all prompts before extension.

No model forward. Uses the frozen prompt helper (small CUDA token transfers).
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from kvcompress.qwen35.runtime import ROOT,kernel_provenance,prompt_tokens
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
import torch
import transformers
from transformers import AutoTokenizer
from kvcompress.qwen35.generate import SamplingConfig,stable_seed
from scripts.qwen35.run_math import source_hashes,atomic_json
from scripts.qwen35.analyze_math import audit_and_load
from scripts.qwen35.analyze_frontier import BASELINE_SHA,data_hash,check_prefix,frozen_sources


def main():
    root=ROOT/'results/qwen35/main32k_b2_v3'
    assert data_hash(root)==BASELINE_SHA
    manifest,records=audit_and_load(root); settings=manifest['settings']; frozen_sources(settings)
    hashes=source_hashes(); path=ROOT/'scripts/qwen35/run_math_batch.py'
    hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashes==settings['source_hashes']
    assert settings['torch']==torch.__version__ and settings['transformers']==transformers.__version__
    assert settings['python']==platform.python_version() and settings['kernels']==kernel_provenance()
    assert settings['sampling']==asdict(SamplingConfig())
    environment=subprocess.check_output(['uv','pip','freeze','--python',str(ROOT/'.venv/bin/python')],text=True)
    assert environment==manifest['environment'],'Installed package environment changed'
    artifacts=json.loads((ROOT/'work/qwen35/artifacts.json').read_text())
    assert artifacts==settings['artifacts']
    data_path=Path(artifacts['dataset_path'])
    assert hashlib.sha256(data_path.read_bytes()).hexdigest()==artifacts['dataset_sha256']
    questions=[json.loads(line) for line in data_path.read_text().splitlines()]
    assert len(questions)==500 and [r['experiment_id'] for r in questions]==settings['problem_ids']
    tokenizer=AutoTokenizer.from_pretrained(artifacts['model_path'],local_files_only=True)
    for question in questions:
        ids=prompt_tokens(tokenizer,question['problem'])[0].tolist()
        for run in (0,1):
            r=records['native'][(question['experiment_id'],run)]
            assert ids==r['prompt_token_ids'] and question['answer']==r['gold']
            for stream in ('generation','eviction'):
                assert stable_seed(question['experiment_id'],run,stream)==r[f'{stream}_seed']
    checked=0
    for name in ('random_pp_C1024','random_pp_C2048'):
        for key,row in records[name].items():
            check_prefix(row,records['native'][key]); checked+=1
    assert not list(root.rglob('*.batch-ungraded'))
    report=dict(status='passed',baseline_data_sha256=BASELINE_SHA,
        baseline_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        source_hashes=hashes,package_environment_exact_match=True,kernels=kernel_provenance(),
        n_baseline_answers_audited=5000,n_current_prompts_matched=500,n_seed_pairs_matched=1000,
        original_compressed_pre_eviction_prefixes_matched=checked,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(ROOT/'docs/plans/qwen35-frontier-preflight.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
