"""Archived evidence check; does not claim model reproduction."""
import importlib.util
from pathlib import Path
import unittest

class PlantedCircuitImport(unittest.TestCase):
    def test_archived_evidence_consistency(self):
        p=Path(__file__).resolve().parents[1]/'experiments/planted_circuits/integration_audit.py'
        spec=importlib.util.spec_from_file_location('planted_audit',p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        result=m.audit()
        self.assertTrue(result['passed'])
        self.assertEqual(result['benchmark_acceptance'],'not_met')
