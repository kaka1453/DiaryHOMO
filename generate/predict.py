import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import build_batch_parser, build_batch_runtime_config
from diary_core.infer.batch import run_generation


def main():
    parser = build_batch_parser()
    args = parser.parse_args()
    runtime = build_batch_runtime_config(args)

    if args.dry_run:
        print("Generate 配置检查通过。")
        print(dump_runtime_config(runtime))
        return

    run_generation(runtime)


if __name__ == "__main__":
    main()
