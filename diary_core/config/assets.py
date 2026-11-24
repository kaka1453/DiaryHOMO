"""Path contract for the product entry points."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from lm_trainer.config.common import asset_path, open_asset_root


RUN_DIRS = ('runs',)


def add_asset_root(parser):
    parser.add_argument('--asset-root', required=True, help='Product asset directory.')


def product_root(value: str) -> Path:
    path = Path(value).expanduser().absolute()
    if any(part.is_symlink() for part in (*path.parents, path)):
        raise ValueError('--asset-root cannot cross a symlink')
    return open_asset_root(str(path))


def external_base_model(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or not path.is_dir():
        raise ValueError('External base model must be an existing absolute directory')
    return path.resolve(strict=True)


def runtime_output(root: Path, value: str | Path, *, directory: bool = False) -> Path:
    """Check both user-selected paths and absolute paths derived during execution."""
    path = Path(value)
    if path.is_absolute():
        if not path.is_relative_to(root):
            raise ValueError('Output path is outside the asset root')
        path = path.relative_to(root)
    return asset_path(root, str(path), 'output', *RUN_DIRS, directory=directory, required=False)


def check_output_tree(root: Path, directory: Path) -> None:
    if not directory.exists():
        return
    for here, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs:
            runtime_output(root, Path(here) / name, directory=True)
        for name in files:
            runtime_output(root, Path(here) / name)


def configure_caches(root: Path) -> None:
    caches = {
        'HF_HOME': 'huggingface',
        'HF_HUB_CACHE': 'hub',
        'HUGGINGFACE_HUB_CACHE': 'hub',
        'TRANSFORMERS_CACHE': 'hub',
        'HF_ASSETS_CACHE': 'hf-assets',
        'HF_XET_CACHE': 'xet',
        'HF_DATASETS_CACHE': 'datasets',
        'HF_MODULES_CACHE': 'modules',
        'MPLCONFIGDIR': 'matplotlib',
        'GRADIO_TEMP_DIR': 'gradio',
        'TORCH_HOME': 'torch',
        'TORCH_EXTENSIONS_DIR': 'torch-extensions',
        'TRITON_CACHE_DIR': 'triton',
        'XDG_CACHE_HOME': 'xdg',
    }
    base = runtime_output(root, 'runs/cache', directory=True)
    check_output_tree(root, base)
    for key, name in caches.items():
        location = runtime_output(root, base / name, directory=True)
        location.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(location)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'


def load_product_config(path: Path, label: str) -> dict:
    from lm_trainer.config.common import load_yaml_config

    try:
        return load_yaml_config(path, label)
    except yaml.YAMLError:
        raise ValueError(f'{label}: invalid YAML') from None


def build_or_exit(parser, builder, args):
    try:
        return builder(args)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    except yaml.YAMLError:
        parser.error('Invalid YAML in product config')
