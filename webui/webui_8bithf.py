import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import build_webui_parser, build_webui_runtime_config
from diary_core.infer.output_bundle import prepare_output_bundle, write_parameters
from diary_core.webui.app import build_demo
from diary_core.webui.state import WebUIState


def main():
    parser = build_webui_parser()
    args = parser.parse_args()
    app_config = build_webui_runtime_config(args)

    if args.dry_run:
        print("WebUI 配置检查通过。")
        print(dump_runtime_config(app_config))
        return

    prepare_output_bundle(app_config)
    write_parameters(app_config)
    Path(app_config["output_run_dir"]).mkdir(parents=True, exist_ok=True)
    print(f"WebUI 输出目录: {app_config['output_run_dir']}")
    demo = build_demo(WebUIState(app_config))
    demo.launch(
        server_name=app_config["server_name"],
        server_port=app_config["server_port"],
        share=app_config["share"],
    )


if __name__ == "__main__":
    main()
