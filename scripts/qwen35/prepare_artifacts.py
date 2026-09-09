"""Fetch and pin public artifacts for the Qwen3.5 experiment (no inference)."""
import hashlib
import json
from pathlib import Path
import subprocess

from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/qwen35"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "artifacts.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    api = HfApi()
    model_id = "Qwen/Qwen3.5-4B"
    dataset_id = "HuggingFaceH4/MATH-500"
    if not manifest:
        manifest = {
            "model_id": model_id,
            "model_revision": api.model_info(model_id).sha,
            "dataset_id": dataset_id,
            "dataset_revision": api.dataset_info(dataset_id).sha,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Pinned artifacts:", manifest, flush=True)
    model_path = snapshot_download(
        model_id, revision=manifest["model_revision"],
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.model", "README.md", "LICENSE"],
        max_workers=4,
    )
    manifest["model_path"] = model_path
    dataset = load_dataset(dataset_id, revision=manifest["dataset_revision"], split="test")
    rows = []
    for i, row in enumerate(dataset):
        row = dict(row)
        row["experiment_id"] = str(row.get("unique_id", i))
        row["source_index"] = i
        rows.append(row)
    assert len(rows) == 500
    assert len({r["experiment_id"] for r in rows}) == 500
    dataset_path = OUT / "math500.jsonl"
    dataset_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    # Hash ranking gives a reproducible split independent of Python RNG/version.
    ordered = sorted(rows, key=lambda r: hashlib.sha256(("qwen35-pilot-v1:" + r["experiment_id"]).encode()).hexdigest())
    pilot_ids = [r["experiment_id"] for r in ordered[:50]]
    (OUT / "pilot_ids.json").write_text(json.dumps(pilot_ids, indent=2) + "\n")
    manifest["dataset_path"] = str(dataset_path)
    manifest["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest["pilot_ids"] = pilot_ids
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    freeze = subprocess.check_output(["uv", "pip", "freeze", "--python", str(ROOT / ".venv/bin/python")], text=True)
    (OUT / "environment.txt").write_text(freeze)
    print("Artifact acquisition complete", manifest_path, flush=True)


if __name__ == "__main__":
    main()
