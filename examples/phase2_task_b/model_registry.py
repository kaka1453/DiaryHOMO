"""Build the BASE and optional final-adapter entries for the local tiny fixture."""

import argparse
import hashlib
from pathlib import Path

import yaml

from lm_trainer.train.artifacts import verify_artifact
from lm_trainer.train.performance import write_json


def write_registry(root, *, include_adapters=False):
    train = yaml.safe_load((root / "direct_sft.yaml").read_text(encoding="utf-8"))
    template = root / "base" / "chat_template.jinja"
    template_hash = hashlib.sha256(
        template.read_text(encoding="utf-8").encode()
    ).hexdigest()
    model = {
        "base_path": "base",
        "context_limit": train["max_length"],
        "fixture_only": True,
        "device": "cpu",
        "dtype": "fp32",
        "quantization": "none",
        "template_policy": {**train["template_policy"], "template_hash": template_hash},
    }
    entries = [{**model, "model_id": "BASE"}]
    if include_adapters:
        for name, run in (("DIRECT_SFT", "direct_sft"), ("MIXED_CPT_SFT", "mixed_sft")):
            relative = f"runs/{run}/final"
            manifest, sha = verify_artifact(root / relative, allow_fixture=True)
            if (
                not manifest["fixture_only"]
                or manifest["template_policy"]["template_hash"] != template_hash
            ):
                raise ValueError(
                    "Expected a verified fixture artifact using the same native template"
                )
            entries.append(
                {
                    **model,
                    "model_id": name,
                    "adapter_path": relative,
                    "manifest_sha256": sha,
                }
            )
    path = root / ("models-ab.json" if include_adapters else "models-base.json")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    write_json(path, {"schema": "lm_trainer.chat-models.v1", "models": entries})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_directory", type=Path)
    parser.add_argument("--include-adapters", action="store_true")
    args = parser.parse_args()
    print(
        write_registry(args.fixture_directory, include_adapters=args.include_adapters)
    )


if __name__ == "__main__":
    main()
