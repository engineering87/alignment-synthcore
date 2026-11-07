"""Testbed 1b metadata. The run itself happens on a GPU in job.py (see its docstring);
import_results.py turns the job's output into records with the standard verdict code."""
from pathlib import Path

from harness.testbed import Testbed

HERE = Path(__file__).resolve().parent


class PlantedLM(Testbed):
    name = "planted_lm"
    version = "0.1"
    safety_metric = "mmcs"  # mean max cosine similarity to the planted directions
    capability_metric = None
    diagnostics = {"child_mmcs": True, "recovery_rate": True}
    evaluator_files = [str(HERE / "job.py"), str(HERE / "testbed.py")]
    public_module = None
    jail_supported = False
    jail_unsupported_reason = "runs on a rented GPU through its own bundle"
    baseline_method = str(HERE / "baseline_topk_sae.py")

    def instance(self, split, config, seed):
        raise RuntimeError("planted_lm runs on a GPU: python -m testbeds.planted_lm.bundle, then import_results")

    def evaluate(self, outputs, inst, seed):
        raise RuntimeError("planted_lm is scored inside job.py")
