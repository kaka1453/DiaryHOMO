import argparse
import json
from lm_trainer.generation import (
    decode_batch,
    generate_batch,
    generation_kwargs,
    normalize_stop_sequences,
    trim_stop_sequences,
)


def extract_ai_reply(text: str) -> str:
    return text.split("AI:")[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation helper smoke test")
    parser.add_argument("--prompt", default="AI: hello")
    args = parser.parse_args()
    print(
        json.dumps(
            {"reply": extract_ai_reply(args.prompt)}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
