"""Resolve inputs and outputs for the historical preprocessing scripts."""

import argparse

from diary_core.config.assets import add_asset_root, build_or_exit, configure_caches, external_base_model, product_root
from lm_trainer.config.common import asset_path


def preprocessing_paths(*, source, target, stats=None, errors=None, tokenizer=False):
    parser = argparse.ArgumentParser(description='Preprocess product data inside the asset root')
    add_asset_root(parser)
    parser.add_argument('--input-file', default=source)
    parser.add_argument('--output-file', default=target)
    if stats is not None:
        parser.add_argument('--stats-file', default=stats)
    if errors is not None:
        parser.add_argument('--error-log', default=errors)
    if tokenizer:
        parser.add_argument('--external-base-model', required=True)

    def resolve(args):
        root = product_root(args.asset_root)
        input_dir = 'data/raw' if source.startswith('data/raw/') else 'data/train'
        paths = [asset_path(root, args.input_file, '--input-file', input_dir)]
        output_dir = 'runs/preprocess' if target.startswith('runs/') else 'data/train'
        paths.append(asset_path(root, args.output_file, '--output-file', output_dir, required=False))
        if stats is not None:
            paths.append(asset_path(root, args.stats_file, '--stats-file', 'runs/preprocess', required=False))
        if errors is not None:
            paths.append(asset_path(root, args.error_log, '--error-log', 'runs/preprocess', required=False))
        if len(set(paths)) != len(paths):
            raise ValueError('Input, output, and diagnostic paths must be distinct')
        model = external_base_model(args.external_base_model) if tokenizer else None
        for path in paths[1:]:
            path.parent.mkdir(parents=True, exist_ok=True)
        if model is not None:
            configure_caches(root)
            paths.append(model)
        return tuple(map(str, paths))

    return build_or_exit(parser, resolve, parser.parse_args())
