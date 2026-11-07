"""Testbed interface. Every testbed implements this so one runner drives all."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Instance:
    """One suite of tasks for one seed.

    public:  list of task dicts the method may see (arrays, weights, sizes).
    private: whatever the evaluator needs to score (ground truth). Never
             written anywhere the method process can read.
    """
    public: list
    private: dict = field(default_factory=dict)
    # Optional public input for a second, separate run of the same method and hparams,
    # made after the main run. Its outputs are stored as private["probe_outputs"]
    # (used by the control testbed's adaptive red team to query the deployed monitor).
    probe: list | None = None


class Testbed:
    name: str = ""
    version: str = ""
    safety_metric: str = ""
    capability_metric: str | None = None
    # diagnostic name -> True if higher means "more of the mechanism"
    diagnostics: dict = {}
    # files whose hash pins the evaluator version
    evaluator_files: list = []
    # file copied into the method sandbox as `public.py`
    public_module: str | None = None
    # Whether methods can run in the container jail (harness/jail.py): no network, no repo.
    jail_supported: bool = True
    jail_unsupported_reason: str = ""
    baseline_method: str = ""

    def instance(self, split: str, config: str, seed: int) -> Instance:
        raise NotImplementedError

    def evaluate(self, outputs: list, inst: Instance, seed: int) -> dict:
        """Return {"safety": float, "capability": float|None, "diagnostics": {...}}"""
        raise NotImplementedError

    def evaluator_hash(self) -> str:
        h = hashlib.sha256()
        for f in sorted(self.evaluator_files):
            h.update(Path(f).name.encode())
            h.update(Path(f).read_bytes())
        return h.hexdigest()[:16]
