"""Config loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import yaml


def load_config(paths: Iterable[str]) -> Dict:
    config: Dict = {}
    for path in paths:
        path_obj = Path(path)
        fragments = _load_with_includes(path_obj)
        for fragment in fragments:
            config = _merge_dicts(config, fragment)
    return config


def _load_with_includes(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        fragment = yaml.safe_load(handle) or {}
    includes = fragment.pop("includes", [])
    fragments: List[Dict] = []
    for include in includes:
        include_path = path.parent / include
        fragments.extend(_load_with_includes(include_path))
    fragments.append(fragment)
    return fragments


def _merge_dicts(base: Dict, update: Dict) -> Dict:
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base
