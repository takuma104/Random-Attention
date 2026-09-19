"""Pin/download the official Qwen3-4B control, isolated from Qwen3.5 artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=ROOT/'work/qwen3_control/artifacts.json')
    args=ap.parse_args(); pin=args.out.with_name('model_pin.json')
    if pin.exists():
        model=json.loads(pin.read_text()); assert model['model_id']=='Qwen/Qwen3-4B'
    else:
        info=HfApi().model_info('Qwen/Qwen3-4B')
        assert not info.private and not info.gated
        model=dict(model_id='Qwen/Qwen3-4B',model_revision=info.sha)
        atomic_json(pin,model)  # preserve the revision even if download is interrupted
    path=Path(snapshot_download(model['model_id'],revision=model['model_revision'],
        allow_patterns=['*.safetensors','*.json','*.jinja','*.txt','*.model','LICENSE','README.md']))
    config=json.loads((path/'config.json').read_text())
    assert config['model_type']=='qwen3'
    files={str(p.relative_to(path)):dict(bytes=p.stat().st_size,sha256=digest(p))
           for p in sorted(path.rglob('*')) if p.is_file() and '.cache' not in p.relative_to(path).parts}
    data=ROOT/'work/qwen35/math500.jsonl'
    original=json.loads((ROOT/'work/qwen35/artifacts.json').read_text())
    assert digest(data)==original['dataset_sha256']
    report=dict(**model,model_path=str(path),config=config,files=files,
        generation_config=json.loads((path/'generation_config.json').read_text()),
        dataset_id=original['dataset_id'],dataset_revision=original['dataset_revision'],
        dataset_path=str(data),dataset_sha256=original['dataset_sha256'],pilot_ids=original['pilot_ids'],
        note='Control preparation only. Original Qwen3 implementation and all Qwen3.5 artifacts unchanged.')
    atomic_json(args.out,report)
    print(json.dumps({k:report[k] for k in ['model_id','model_revision','model_path','config','generation_config']},indent=2))
    print('Downloaded and hashed bytes:',sum(f['bytes'] for f in files.values()))


if __name__=='__main__': main()
