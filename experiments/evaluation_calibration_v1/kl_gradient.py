"""Exact-gradient contracts for explicitly declared KL toy implementations.

These adapters are authored here. They do not import or certify the archived KL
anchoring training code. Enumeration removes Monte Carlo sampling uncertainty.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

TOLERANCE = 1e-7
EPSILON = 1e-6
VARIANTS = {
    "reverse_exact": ("reverse", True),
    "reverse_score_surrogate": ("reverse", True),
    "forward_exact": ("forward", True),
    "k3_detached_sampling_forward": ("forward", True),
    "k1_detached_sampling_as_reverse": ("reverse", False),
    "k3_detached_sampling_misdeclared_reverse": ("reverse", False),
    "fully_detached_reverse": ("reverse", False),
    "misnormalized_reverse": ("reverse", False),
}


def protocol_spec() -> dict:
    return {
        "name": "enumerated-kl-gradient-contract-v1",
        "scope": "New explicit toy implementations; does not validate archived KL training code or establish alignment behavior.",
        "cases": "3 nonstationary categorical 3-action fixtures and 3 deterministic two-state, two-action, horizon-3 MDP fixtures",
        "reference_support": "strictly positive fixed reference policy; current policy softmax over float64 logits",
        "objectives": {"reverse": "sum_trajectories p * log(p/q)", "forward": "sum_trajectories q * log(q/p)"},
        "sampling_contract": "Enumerate all categorical actions or all 2^3 trajectories. For sampled-loss variants, current-policy probability weights are detached; score-surrogate also detaches the log-ratio coefficient. Gradients are exact expectations, not individual sample gradients.",
        "k3_contract": "k3=q/p-log(q/p)-1; direct loss with detached p sampling weights has the forward-KL gradient in this full-trajectory construction, despite its reverse-KL expected value.",
        "mdp": "start state 0; action 0 preserves state, action 1 flips state; full trajectory log probabilities include all 3 decisions; fixed reference uses same dynamics",
        "variants": {k: {"declared_objective": objective, "expected_accept": valid} for k, (objective, valid) in VARIANTS.items()},
        "negative_controls": "k1 direct sampled loss lacks the score-distribution term; k3 deliberately assigned wrong objective; fully detached gradient; mean reduction substituted for declared sum",
        "finite_difference": {"method": "central difference of separately evaluated NumPy objective", "epsilon": EPSILON, "max_absolute_error_tolerance": TOLERANCE},
        "categorical_analytic_crosscheck": "reverse: p*(log(p/q)-KL); forward: p-q",
        "stopping": "All 6 cases and 8 variants, fixed tolerances and no result-driven modifications; engineering unit checks may use other fixtures before freeze.",
    }


def _softmax(x):
    z = x - np.max(x, axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def case_specifications():
    return [
        {"id": "categorical-1", "kind": "categorical", "logits": [.7, -.2, 1.3], "reference": [.2, .5, .3]},
        {"id": "categorical-2", "kind": "categorical", "logits": [-1., .3, .1], "reference": [.6, .3, .1]},
        {"id": "categorical-3", "kind": "categorical", "logits": [.5, .1, -.4], "reference": [.15, .25, .6]},
        {"id": "mdp-1", "kind": "mdp", "horizon": 3, "logits": [[.4, -.7], [-.2, .9]], "reference": [[.7, .3], [.2, .8]]},
        {"id": "mdp-2", "kind": "mdp", "horizon": 3, "logits": [[-.5, .8], [.7, -.3]], "reference": [[.8, .2], [.35, .65]]},
        {"id": "mdp-3", "kind": "mdp", "horizon": 3, "logits": [[1., .2], [-.6, .4]], "reference": [[.4, .6], [.75, .25]]},
    ]


def _numpy_probabilities(logits: np.ndarray, case: dict):
    policy = _softmax(logits); reference = np.asarray(case["reference"], float)
    if case["kind"] == "categorical":
        return policy, reference
    ps, qs = [], []
    for sequence in itertools.product(range(2), repeat=case["horizon"]):
        state = 0; p = 1.; q = 1.
        for action in sequence:
            p *= policy[state, action]; q *= reference[state, action]
            state = (state + action) % 2
        ps.append(p); qs.append(q)
    return np.asarray(ps), np.asarray(qs)


def numpy_objective(logits: np.ndarray, case: dict, objective: str) -> float:
    p, q = _numpy_probabilities(logits, case)
    return float(np.sum(p * np.log(p / q)) if objective == "reverse" else np.sum(q * np.log(q / p)))


def finite_difference(case: dict, objective: str) -> np.ndarray:
    x = np.asarray(case["logits"], float); grad = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus = x.copy(); minus = x.copy(); plus[index] += EPSILON; minus[index] -= EPSILON
        grad[index] = (numpy_objective(plus, case, objective) - numpy_objective(minus, case, objective)) / (2 * EPSILON)
    return grad


def _torch_log_probabilities(theta, case):
    policy = torch.log_softmax(theta, dim=-1)
    reference = torch.as_tensor(case["reference"], dtype=torch.float64).log()
    if case["kind"] == "categorical":
        return policy, reference
    ps, qs = [], []
    for sequence in itertools.product(range(2), repeat=case["horizon"]):
        state = 0; lp = theta.sum() * 0; lq = reference.sum() * 0
        for action in sequence:
            lp = lp + policy[state, action]; lq = lq + reference[state, action]
            state = (state + action) % 2
        ps.append(lp); qs.append(lq)
    return torch.stack(ps), torch.stack(qs)


def variant_gradient(case: dict, variant: str) -> np.ndarray:
    theta = torch.tensor(case["logits"], dtype=torch.float64, requires_grad=True)
    logp, logq = _torch_log_probabilities(theta, case)
    p, q = logp.exp(), logq.exp(); ratio_log = logp - logq
    if variant == "reverse_exact":
        loss = (p * ratio_log).sum()
    elif variant == "reverse_score_surrogate":
        loss = (p.detach() * ratio_log.detach() * logp).sum()
    elif variant == "forward_exact":
        loss = (q * -ratio_log).sum()
    elif variant in ("k3_detached_sampling_forward", "k3_detached_sampling_misdeclared_reverse"):
        q_over_p = (logq - logp).exp()
        loss = (p.detach() * (q_over_p - (logq - logp) - 1)).sum()
    elif variant == "k1_detached_sampling_as_reverse":
        loss = (p.detach() * ratio_log).sum()
    elif variant == "fully_detached_reverse":
        loss = (p * ratio_log).sum().detach() + theta.sum() * 0
    elif variant == "misnormalized_reverse":
        # The declared objective is a sum, not an average over action entries/time.
        divisor = len(case["reference"]) if case["kind"] == "categorical" else case["horizon"]
        loss = (p * ratio_log).sum() / divisor
    else:
        raise ValueError(variant)
    return torch.autograd.grad(loss, theta)[0].detach().numpy()


def evaluate_case(case: dict) -> dict:
    reference = {objective: finite_difference(case, objective) for objective in ("reverse", "forward")}
    rows = []
    for variant, (objective, expected) in VARIANTS.items():
        gradient = variant_gradient(case, variant)
        error = float(np.max(np.abs(gradient - reference[objective])))
        accepted = error <= TOLERANCE
        rows.append({"variant": variant, "declared_objective": objective, "expected_accept": expected,
                     "accepted": accepted, "contract_check_correct": accepted == expected,
                     "max_absolute_error": error, "gradient": gradient.tolist(),
                     "finite_difference_gradient": reference[objective].tolist()})
    analytic = None
    if case["kind"] == "categorical":
        p, q = _numpy_probabilities(np.asarray(case["logits"], float), case)
        logratio = np.log(p / q)
        gradients = {"reverse": p * (logratio - np.sum(p * logratio)), "forward": p - q}
        analytic = {name: {"gradient": value.tolist(),
                           "max_absolute_error": float(np.max(np.abs(value - reference[name]))),
                           "passed": bool(np.max(np.abs(value - reference[name])) <= TOLERANCE)}
                    for name, value in gradients.items()}
    p, q = _numpy_probabilities(np.asarray(case["logits"], float), case)
    return {"case": case, "current_probability_sum": float(p.sum()), "reference_probability_sum": float(q.sum()),
            "current_probabilities": p.tolist(), "reference_probabilities": q.tolist(),
            "rows": rows, "analytic_crosscheck": analytic}


def run(output_dir: Path) -> dict:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(json.dumps(protocol_spec(), indent=2) + "\n")
    records = [evaluate_case(case) for case in case_specifications()]
    rows = [row for record in records for row in record["rows"]]
    summary = {"study": protocol_spec()["name"], "cases": len(records), "contracts": len(rows),
               "correct_accept_reject": sum(row["contract_check_correct"] for row in rows),
               "positive_contracts_accepted": sum(row["accepted"] and row["expected_accept"] for row in rows),
               "negative_controls_rejected": sum(not row["accepted"] and not row["expected_accept"] for row in rows),
               "all_probability_sums_valid": all(abs(r["current_probability_sum"] - 1) < 1e-12 and abs(r["reference_probability_sum"] - 1) < 1e-12 for r in records),
               "all_analytic_crosschecks_passed": all(v["passed"] for r in records if r["analytic_crosscheck"] for v in r["analytic_crosscheck"].values()),
               "scope": protocol_spec()["scope"]}
    (output_dir / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
