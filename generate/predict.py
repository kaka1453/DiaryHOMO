import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "generate.yaml"


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="8bit HuggingFace batch generation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config file path.",
    )
    parser.add_argument(
        "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default=None,
        help="Override model path from config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        type=str,
        default=None,
        help="Override LoRA checkpoint path from config.",
    )
    parser.add_argument(
        "--input-file",
        dest="input_file",
        type=str,
        default=None,
        help="Override input prompt file path from config.",
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        type=str,
        default=None,
        help="Override markdown output file path from config.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=None,
        help="Override inference device from config.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="Override max_new_tokens from config.",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=None,
        help="Override temperature from config.",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Override top_p from config.",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Override top_k from config.",
    )
    parser.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=None,
        help="Override repetition_penalty from config.",
    )
    parser.add_argument(
        "--num-beams",
        dest="num_beams",
        type=int,
        default=None,
        help="Override num_beams from config.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only validate config and print resolved values without loading the model.",
    )
    parser.add_argument(
        "--print-prompts",
        dest="print_prompts",
        type=str2bool,
        default=None,
        help="Override whether prompts are printed before generation.",
    )
    return parser


def load_yaml_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("config/generate.yaml 顶层必须是映射对象。")
    return data


def resolve_path(value: str):
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_runtime_config(args):
    config_path = Path(args.config).expanduser().resolve()
    raw_config = load_yaml_config(config_path)
    generation_cfg = raw_config.get("generation") or {}

    runtime = {
        "model_name_or_path": raw_config.get("model_name_or_path"),
        "checkpoint_dir": raw_config.get("checkpoint_dir"),
        "input_file": raw_config.get("input_file"),
        "output_file": raw_config.get("output_file"),
        "device": raw_config.get("device"),
        "batch_size": raw_config.get("batch_size"),
        "print_prompts": raw_config.get("print_prompts", True),
        "max_new_tokens": generation_cfg.get("max_new_tokens"),
        "temperature": generation_cfg.get("temperature"),
        "top_p": generation_cfg.get("top_p"),
        "top_k": generation_cfg.get("top_k"),
        "repetition_penalty": generation_cfg.get("repetition_penalty"),
        "num_beams": generation_cfg.get("num_beams"),
    }

    cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "dry_run"} and value is not None
    }
    runtime.update(cli_overrides)

    required_keys = [
        "model_name_or_path",
        "checkpoint_dir",
        "input_file",
        "output_file",
        "device",
        "batch_size",
        "print_prompts",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "num_beams",
    ]
    missing = [key for key in required_keys if runtime.get(key) is None]
    if missing:
        raise ValueError(f"generate 配置缺少字段: {', '.join(missing)}")

    runtime["config_path"] = config_path
    runtime["model_name_or_path"] = str(resolve_path(runtime["model_name_or_path"]))
    runtime["checkpoint_dir"] = str(resolve_path(runtime["checkpoint_dir"]))
    runtime["input_file"] = str(resolve_path(runtime["input_file"]))
    runtime["output_file"] = str(resolve_path(runtime["output_file"]))
    runtime["batch_size"] = int(runtime["batch_size"])
    runtime["print_prompts"] = bool(runtime["print_prompts"])
    runtime["max_new_tokens"] = int(runtime["max_new_tokens"])
    runtime["temperature"] = float(runtime["temperature"])
    runtime["top_p"] = float(runtime["top_p"])
    runtime["top_k"] = int(runtime["top_k"])
    runtime["repetition_penalty"] = float(runtime["repetition_penalty"])
    runtime["num_beams"] = int(runtime["num_beams"])
    return runtime


def dump_runtime_config(runtime):
    printable = dict(runtime)
    printable["config_path"] = str(printable["config_path"])
    return json.dumps(printable, ensure_ascii=False, indent=2)


def load_prompts(input_file: str):
    with open(input_file, "r", encoding="utf-8") as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    if not prompts:
        raise ValueError(f"输入文件中未读取到有效 prompt: {input_file}")
    return prompts


def load_model_and_tokenizer(runtime):
    tokenizer = AutoTokenizer.from_pretrained(
        runtime["model_name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        runtime["model_name_or_path"],
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, runtime["checkpoint_dir"])
    model.eval()
    return tokenizer, model


def generate_batch(prompts, tokenizer, model, runtime):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(runtime["device"])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=runtime["max_new_tokens"],
            temperature=runtime["temperature"],
            top_p=runtime["top_p"],
            top_k=runtime["top_k"],
            repetition_penalty=runtime["repetition_penalty"],
            do_sample=True,
            num_beams=runtime["num_beams"],
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def write_results(results, output_file: str):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.writelines(results)


def run_generation(runtime):
    torch.backends.cuda.matmul.allow_tf32 = True

    prompts = load_prompts(runtime["input_file"])
    if runtime["print_prompts"]:
        print(f"读取到 {len(prompts)} 条 prompts。")

    tokenizer, model = load_model_and_tokenizer(runtime)

    results = []
    batch_size = runtime["batch_size"]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        texts = generate_batch(batch, tokenizer, model, runtime)

        for prompt, text in zip(batch, texts):
            idx = len(results) + 1
            block = (
                f"## 第{idx}篇\n\n"
                f"引言: {prompt}\n\n"
                f"{text}\n\n"
                f"---\n"
            )
            if runtime["print_prompts"]:
                print(block)
            results.append(block)

    write_results(results, runtime["output_file"])
    print(f"\n✅ 8bit 推理完成，共 {len(results)} 篇，已写入 {runtime['output_file']}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    runtime = build_runtime_config(args)

    if args.dry_run:
        print("Generate 配置检查通过。")
        print(dump_runtime_config(runtime))
        return

    run_generation(runtime)


if __name__ == "__main__":
    main()
