"""Load product wording and rules as data and bind them to a single generation."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path

from lm_trainer.config.common import asset_path


CONTENT_FILES = {
    'diary_profile': ('assets/persona', 'matching.json'),
    'prompt_builder': ('assets/prompts', 'builder.json'),
    'prompt_contract': ('assets/prompts', 'contract.json'),
    'diary_runtime': ('assets/prompts', 'retry.json'),
    'guard': ('assets/guard', 'rules.json'),
}
_CONTENT = ContextVar('product_content', default=None)


def load_product_content(root: Path) -> dict:
    content = {}
    for section, (directory, name) in CONTENT_FILES.items():
        path = asset_path(root, f'{directory}/{name}', section, directory)
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
        except (UnicodeError, ValueError):
            raise ValueError(f'{section}: invalid JSON in {name}') from None
        if not isinstance(result, dict) or not isinstance(result.get('values'), dict):
            raise ValueError(f'{section}: expected a values mapping in {name}')
        wording = result.get('wording')
        if not isinstance(wording, dict) or any(
            not isinstance(group, list) or any(not isinstance(item, str) for item in group)
            for group in wording.values()
        ):
            raise ValueError(f'{section}: expected a wording mapping in {name}')
        content[section] = result
    return content


@contextmanager
def using_product_content(content: dict):
    token = _CONTENT.set(content)
    try:
        yield
    finally:
        _CONTENT.reset(token)


def product_value(section: str, name: str):
    content = _CONTENT.get()
    try:
        return content[section]['values'][name]
    except (KeyError, TypeError):
        raise ValueError(f'Missing product setting {section}.{name}') from None


def product_wording(section: str, group: str, index: int) -> str:
    content = _CONTENT.get()
    try:
        return content[section]['wording'][group][index]
    except (IndexError, KeyError, TypeError):
        raise ValueError(f'Missing product wording {section}.{group}[{index}]') from None
