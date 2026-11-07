"""Capabilities: what testbeds provide and what untestable ideas need (capabilities.json).

This is the learning edge's vocabulary. An untestable verdict names its missing
capabilities; a testbed that provides all of them makes the verdict stale.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "capabilities.json"


def load() -> dict:
    return json.loads(FILE.read_text())


def covering_testbed(needs, data: dict | None = None) -> str | None:
    """The first testbed that provides every need, or None. 'other' is never provided."""
    data = data or load()
    needs = set(needs or [])
    if not needs or "other" in needs:
        return None
    for tb, provides in sorted(data["provides"].items()):
        if needs <= set(provides):
            return tb
    return None
