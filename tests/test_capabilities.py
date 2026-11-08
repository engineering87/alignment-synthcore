"""The capability vocabulary stays consistent, and no untestable verdict goes stale silently."""
import json
import re
from pathlib import Path

import pytest

from harness import capabilities

ROOT = Path(__file__).resolve().parent.parent
DATA = capabilities.load()
VOCAB = DATA["capabilities"]
IDEAS = {p.parent.name: json.loads(p.read_text()) for p in sorted((ROOT / "ideas").glob("*/idea.json"))}
# Every name the harness can load, including variants that share a testbed directory
# (swarm_collusion_para lives in testbeds/swarm_collusion/).
TESTBEDS = sorted(set(re.findall(r'if name == "(\w+)"', (ROOT / "harness" / "runner.py").read_text())))


def test_vocabulary_is_well_formed():
    assert "other" in VOCAB
    for cid, c in VOCAB.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]{1,40}", cid), cid
        assert c["name"].strip() and len(c["description"]) >= 20, cid


def test_every_testbed_declares_what_it_provides():
    assert sorted(DATA["provides"]) == TESTBEDS
    for tb, provides in DATA["provides"].items():
        assert provides and set(provides) <= set(VOCAB) - {"other"}, tb


@pytest.mark.parametrize("slug", [s for s, i in IDEAS.items() if i.get("testbed") is None])
def test_untestable_ideas_name_missing_capabilities(slug):
    idea = IDEAS[slug]
    needs = idea.get("needs")
    assert needs and set(needs) <= set(VOCAB), f"{slug}: needs must use capabilities.json ids"
    assert idea.get("untestable_reason"), slug
    tb = capabilities.covering_testbed(needs, DATA)
    assert tb is None, (f"{slug} is stale: testbed {tb} provides {needs}. "
                        "Give it that testbed and a prediction, or name the capability it still lacks.")


def test_a_new_testbed_reopens_matching_verdicts():
    data = json.loads(json.dumps(DATA))
    assert capabilities.covering_testbed(["poisoned_training_data"], data) is None
    data["provides"]["poisoning"] = ["model_weights", "poisoned_training_data"]
    assert capabilities.covering_testbed(["poisoned_training_data"], data) == "poisoning"
    assert capabilities.covering_testbed(["poisoned_training_data", "gpu"], data) is None  # one testbed must cover all
    assert capabilities.covering_testbed(["other"], data) is None


# Four verdicts that were once untestable and were resolved by building the testbed
# that supplied the missing capability. Tagged as their reasons read, each would have
# reopened automatically the day that testbed landed.
HISTORY = {
    "decoupled-approval": (["online_interaction", "approval_queries"], "interactive_gridworld"),
    "mona": (["online_interaction", "approval_queries"], "interactive_gridworld"),
    "minimax-regret-levels": (["online_interaction", "level_generation"], "ued_gridworld"),
    "causal-confusion-interventions": (["online_interaction", "expert_queries"], "confounded_imitation"),
}


@pytest.mark.parametrize("slug", sorted(HISTORY))
def test_past_untestable_verdicts_would_have_reopened(slug):
    needs, built = HISTORY[slug]
    assert capabilities.covering_testbed(needs, DATA) == built
    assert IDEAS[slug]["testbed"] == built
