"""Single-worker loopback launcher; secrets are read from operator-named env vars."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from uuid import uuid4

from lm_trainer.chat import (
    ChatError,
    ChatRuntime,
    GenerationSettings,
    load_registry,
    load_roles,
)
from lm_trainer.chat.inference import (
    InferenceServiceV1,
    ScopeBindingV1,
    error_status,
    write_audit,
)
from lm_trainer.train.artifacts import read_json

from .app import APIGrantV1, create_app


def _config(path, *, test_mode):
    try:
        config = read_json(path)
        if config.get("schema") != "lm_trainer.inference-service.v1" or set(config) - {
            "schema",
            "registry",
            "roles",
            "model_id",
            "scopes",
            "credentials",
            "generation",
            "max_queued",
            "timeout_seconds",
            "max_output_tokens",
            "max_body_bytes",
            "body_timeout_seconds",
        }:
            raise ValueError
        registry = load_registry(
            path.parent / config["registry"], allow_fixture=test_mode
        )
        default = GenerationSettings(**config.get("generation", {}))
        service = InferenceServiceV1(
            ChatRuntime(registry, load_roles(path.parent / config["roles"])),
            config["model_id"],
            tuple(ScopeBindingV1(**entry) for entry in config["scopes"]),
            defaults=default,
            test_mode=test_mode,
            **{
                k: config[k]
                for k in ("max_queued", "timeout_seconds", "max_output_tokens")
                if k in config
            },
        )
        credentials = {}
        for entry in config["credentials"]:
            if set(entry) != {
                "token_env",
                "principal_id",
                "allowed_models",
                "allowed_scopes",
            }:
                raise ValueError
            token = os.environ[entry["token_env"]]
            if token in credentials or any(
                not isinstance(entry[k], list)
                for k in ("allowed_models", "allowed_scopes")
            ):
                raise ValueError
            credentials[token] = APIGrantV1(
                entry["principal_id"],
                frozenset(entry["allowed_models"]),
                frozenset(entry["allowed_scopes"]),
            )
        app = create_app(
            service,
            credentials,
            **{
                k: config[k]
                for k in ("max_body_bytes", "body_timeout_seconds")
                if k in config
            },
        )
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        raise ChatError("INVALID_CONFIG") from None
    return service, app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify, warm up and unload; never open a socket",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="explicitly allow labeled local fixtures only for testing",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    # Our logger emits allowlisted metadata. Avoid source paths in third-party model
    # diagnostics and credentials or scope in an access log. No change on import.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in (
        "transformers",
        "torch",
        "peft",
        "huggingface_hub",
        "safetensors",
        "accelerate",
    ):
        logging.getLogger(name).setLevel(logging.CRITICAL + 1)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    try:
        service, app = _config(args.config, test_mode=args.test_mode)
        # Fail before binding a port if startup cannot activate the pinned model.
        service.start()
        try:
            if args.check:
                write_audit(
                    request_id=uuid4().hex,
                    status=200,
                    error_code=None,
                    model_id=service.model_id,
                )
            else:
                import uvicorn

                uvicorn.run(
                    app,
                    host="127.0.0.1",
                    port=args.port,
                    workers=1,
                    access_log=False,
                    proxy_headers=False,
                )
        finally:
            service.close()
    except ChatError as exc:
        write_audit(
            request_id=uuid4().hex, status=error_status(exc.code), error_code=exc.code
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
