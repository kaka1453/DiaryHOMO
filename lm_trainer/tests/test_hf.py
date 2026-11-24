import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from lm_trainer.train.engine import accumulation_windows, save_adapter, update_window


class ScalarLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, labels):
        targets = labels[:, 1:].flatten()
        logits = torch.stack([self.weight * 0, self.weight]).expand(len(targets), -1)
        return SimpleNamespace(loss=cross_entropy(logits, targets))


def test_real_loader_matches_large_batch_and_flushes_tail():
    rows = [[0, 1, -100, -100], [1, 0, 0, 0], [0, 1, 1, -100]]
    loader = DataLoader([{"labels": torch.tensor(r)} for r in rows], batch_size=1)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
    model, reference = ScalarLM(), ScalarLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1)
    for index, window in enumerate(accumulation_windows(loader, 2)):
        # Compare with one forward/backward over the concatenated large batch.
        reference(torch.cat([b["labels"] for b in window])).loss.backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad()
        _, tokens = update_window(model, optimizer, scheduler, accelerator, window)
        torch.testing.assert_close(model.weight, reference.weight)
        assert model.weight.item() == pytest.approx([-0.25, 0.3121765009][index])
        assert tokens == [4, 2][index]
    assert scheduler.last_epoch == 2
    before = model.weight.detach().clone()
    empty = {"labels": torch.tensor([[1, -100]])}
    with pytest.raises(ValueError, match="no next-token"):
        update_window(model, optimizer, scheduler, accelerator, [empty])
    torch.testing.assert_close(model.weight, before)
    assert scheduler.last_epoch == 2
    update_window(
        model,
        optimizer,
        scheduler,
        accelerator,
        [empty, {"labels": torch.tensor([[0, 1]])}],
    )
    assert scheduler.last_epoch == 3


def test_incomplete_export_is_never_published(tmp_path):
    def fail(*args, **kwargs):
        raise OSError("simulated save failure")

    with pytest.raises(OSError, match="simulated"):
        save_adapter(tmp_path / "final", SimpleNamespace(save_pretrained=fail), None, 1)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("save_every", [20, 1])
def test_hf_cli_synthetic_adapter_round_trip(tmp_path, save_every):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    base, output = tmp_path / "base", tmp_path / "out"
    vocab = {
        ch: i for i, ch in enumerate(["<pad>", "<unk>", *"onetwhrĠ", "<|endoftext|>"])
    }
    raw = Tokenizer(BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>"
    )
    tokenizer.save_pretrained(base)
    raw.model.save(str(base))
    config_file = base / "tokenizer_config.json"
    config_file.write_text(
        json.dumps(
            {**json.loads(config_file.read_text()), "tokenizer_class": "Qwen2Tokenizer"}
        )
    )
    torch.manual_seed(3)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            eos_token_id=1,
            pad_token_id=0,
        )
    ).save_pretrained(base)
    data = tmp_path / "fixture.jsonl"
    data.write_text(
        "".join(
            json.dumps({"text": text}) + "\n"
            for text in ("one two", "two three one two", "three two one")
        )
    )
    root = Path(__file__).resolve().parents[2]
    train_config, model_config = tmp_path / "train.yaml", tmp_path / "model.yaml"
    train_config.write_text(
        json.dumps({"shuffle": False, "auto_pad_batch": True, "seed": 3})
    )
    model_config.write_text(json.dumps({"lora_bias": "none"}))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "ACCELERATE_USE_CPU": "true",
        "HF_HOME": str(tmp_path / "hf"),
        "HF_DATASETS_CACHE": str(tmp_path / "datasets"),
        "HF_HUB_OFFLINE": "1",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    cmd = [
        sys.executable,
        "-m",
        "lm_trainer",
        "--config",
        str(train_config),
        "--model-config",
        str(model_config),
        "--model-name-or-path",
        str(base),
        "--checkpoint-dir",
        str(output),
        "--data-file-path",
        str(data),
        "--quantization-mode",
        "none",
        "--lora-target-modules",
        "q_proj,v_proj",
        "--max-length",
        "16",
        "--lora-r",
        "2",
        "--lora-alpha",
        "4",
        "--lora-dropout",
        "0",
        "--lr",
        "0.02",
        "--batch-size",
        "1",
        "--gradient-accumulation-steps",
        "2",
        "--epochs",
        "1",
        "--max-steps",
        "9",
        "--warmup-steps",
        "0",
        "--save-steps",
        str(save_every),
        "--num-workers",
        "0",
        "--pin-memory",
        "false",
    ]
    subprocess.run(cmd, cwd=root, env=env, check=True)
    saved_steps = sorted(path.name for path in output.glob("step_*"))
    assert saved_steps == (["step_1", "step_2"] if save_every == 1 else [])
    report = json.loads((output / "performance.json").read_text())
    assert report["optimizer_updates"] == 2 and report["backend"] == "hf"
    assert [r["microbatches"] for r in report["updates"]] == [2, 1]
    assert report["device"] == "cpu" and report["peak_allocated_bytes"] is None
    assert report["lock_sha256"] == {}
    assert all(key.startswith("lm_trainer/") for key in report["source_sha256"])
    manifest = json.loads((output / "final/training_manifest.json").read_text())
    assert manifest == {"artifact_kind": "inference_adapter", "optimizer_updates": 2}
    check_reload = """
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, adapter = sys.argv[1:]
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), adapter).eval()
x = AutoTokenizer.from_pretrained(adapter)('one two', return_tensors='pt')['input_ids']
with torch.no_grad():
    with model.disable_adapter():
        off = model(x).logits
    on = model(x).logits
    with model.disable_adapter():
        off_again = model(x).logits
    torch.testing.assert_close(off, off_again, rtol=0, atol=0)
    assert torch.isfinite(on).all() and (on - off).abs().max().item() > 0
"""
    subprocess.run(
        [sys.executable, "-c", check_reload, str(base), str(output / "final")],
        env=env,
        check=True,
    )
