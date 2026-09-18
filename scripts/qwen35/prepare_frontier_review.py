"""Reuse the generic packet selector with frontier-specific exposure disclosure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_grading_review.py')),
        str(args.root),'--out',str(args.out)],check=True,capture_output=True,text=True)
    path=args.out/'manifest.json'; meta=json.loads(path.read_text())
    meta['prior_exposure']='Original main-study review and aggregate frontier statistics were seen. New frontier packets hide condition/run; recurring question/answer pairs may be familiar. Not independent human adjudication.'
    path.write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))


if __name__=='__main__': main()
