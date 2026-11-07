"""Paired AUROC uncertainty calibration with explicit sampling units.

Score layout is [detector_seed, victim_model, arm], with baseline arm 0 and
candidate arm 1. Victims and detector seeds are crossed: resampling a detector
seed retains its complete score vector and the same paired arm observations.
Bootstrap draws are Monte Carlo replicates, never experimental observations.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.stats import t

VERSION = "paired-auroc-estimands-v1"
MODES = ("fixed_victims_new_detectors", "new_victims_fixed_detectors", "new_victims_new_detectors")


def _validated(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 3 or scores.shape[2] != 2 or scores.shape[0] < 1:
        raise ValueError("scores must have shape [detectors, victims, 2]")
    if labels.ndim != 1 or len(labels) != scores.shape[1] or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("binary labels must match victims")
    if not (np.any(labels == 0) and np.any(labels == 1)) or not np.isfinite(scores).all():
        raise ValueError("both classes and finite scores required")
    return scores, labels.astype(int)


def _arm_kernels(scores, labels):
    scores, labels = _validated(scores, labels)
    pos, neg = scores[:, labels == 1, :], scores[:, labels == 0, :]
    delta = pos[:, :, None, :] - neg[:, None, :, :]
    return (delta > 0).astype(float) + 0.5 * (delta == 0)


def paired_auroc(scores, labels) -> dict:
    kernels = _arm_kernels(scores, labels)
    values = kernels.mean(axis=(1, 2))
    effects = values[:, 1] - values[:, 0]
    return {"baseline": float(values[:, 0].mean()), "candidate": float(values[:, 1].mean()),
            "effect": float(effects.mean()), "per_detector": values.tolist(), "per_detector_effect": effects.tolist()}


def bootstrap_effects(scores, labels, draws: int, seed: int, mode: str) -> np.ndarray:
    """Paired stratified victim bootstrap, with optional whole-seed resampling.

    new_victims_new_detectors is a crossed bootstrap: victim counts are shared
    by all detector seeds, and detector counts are shared by all victims.
    This retains both arms and the paired design in every resample. Its finite
    sample coverage is measured below rather than assumed to be exact.
    """
    if mode not in MODES or isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("valid mode and at least two bootstrap draws required")
    kernels = _arm_kernels(scores, labels)
    difference = kernels[..., 1] - kernels[..., 0]
    detectors, positive, negative = difference.shape
    if mode != "new_victims_fixed_detectors" and detectors < 2:
        raise ValueError("detector uncertainty requires at least two independent detector seeds")
    rng = np.random.default_rng(seed)
    if mode == "fixed_victims_new_detectors":
        counts = rng.multinomial(detectors, np.full(detectors, 1 / detectors), size=draws) / detectors
        return counts @ difference.mean(axis=(1, 2))
    positive_counts = rng.multinomial(positive, np.full(positive, 1 / positive), size=draws) / positive
    negative_counts = rng.multinomial(negative, np.full(negative, 1 / negative), size=draws) / negative
    if mode == "new_victims_fixed_detectors":
        return ((positive_counts @ difference.mean(axis=0)) * negative_counts).sum(axis=1)
    detector_counts = rng.multinomial(detectors, np.full(detectors, 1 / detectors), size=draws) / detectors
    result = np.zeros(draws)
    for detector in range(detectors):
        auc = ((positive_counts @ difference[detector]) * negative_counts).sum(axis=1)
        result += detector_counts[:, detector] * auc
    return result


def percentile_interval(draws) -> list[float]:
    values = np.asarray(draws, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("at least two finite bootstrap draws required")
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def wilson_interval(successes: int, trials: int) -> list[float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("valid binomial counts required")
    z = 1.959963984540054
    p, denominator = successes / trials, 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [float(center - margin), float(center + margin)]


def _normal_scores(rng, models_per_class, detectors, mu_base, mu_candidate, detector_noise=0.0):
    labels = np.repeat([0, 1], models_per_class)
    common = rng.normal(size=len(labels))
    independent = rng.normal(size=(len(labels), 2))
    latent = np.sqrt(0.6) * common[:, None] + np.sqrt(0.4) * independent
    latent += labels[:, None] * np.array([mu_base, mu_candidate])[None, :]
    scores = latent[None, :, :] + detector_noise * rng.normal(size=(detectors, len(labels), 2))
    variance = 1 + detector_noise ** 2
    truth = float(ndtr(mu_candidate / np.sqrt(2 * variance)) - ndtr(mu_base / np.sqrt(2 * variance)))
    return scores, labels, truth, latent


def protocol_spec() -> dict:
    return {"id": "uncertainty-calibration-v1", "version": VERSION, "stage": "synthetic_calibration",
            "estimands": {
                "fixed_victims_new_detectors": "expected paired AUROC difference across new independent detector fits on this exact victim corpus",
                "new_victims_fixed_detectors": "paired AUROC difference on new victims for the average of these fixed detector fits",
                "new_victims_new_detectors": "expected paired AUROC difference over new victim models and independent detector fits"},
            "bootstrap": {"interval": "95% paired percentile", "victims": "stratified by clean/backdoored class",
                          "detectors": "resample whole detector score vectors; preserve arm pairing; never independent per-victim seeds",
                          "coverage_draws": 399, "stability_draw_prefixes": [128, 512, 2048]},
            "stability_table": {"data_seed": 2026410500, "bootstrap_seed": 2026410501,
                                "models_per_class": 80, "detector_seeds": 1, "mu_baseline": 0.7, "mu_candidate": 1.0,
                                "detector_noise_sd": 0.0, "arm_latent_correlation": 0.6},
            "population_simulations": {"repetitions_per_condition": 200, "models_per_class": 80,
                                       "conditions": ["fixed_detector_null", "fixed_detector_effect", "crossed_null", "crossed_effect"],
                                       "fixed_detector_seeds": 1, "crossed_detector_seeds": 4,
                                       "mu_baseline": 0.7, "mu_candidate_null": 0.7, "mu_candidate_effect": 1.0,
                                       "arm_latent_correlation": 0.6, "crossed_detector_noise_sd": 0.3,
                                       "seed_base": 2026110200},
            "fixed_victim_simulation": {"repetitions": 200, "models_per_class": 80, "detector_seeds": 12,
                                        "detector_noise_sd": 0.5, "latent_seed": 2026210300, "draw_seed_base": 2026310400,
                                        "conditional_truth": "mean over victim pairs of Gaussian comparison probabilities; fixed latent corpus"},
            "criteria": {
                "coverage_each": "empirical coverage >=0.90 and two-sided 95% Wilson interval contains nominal 0.95",
                "stability": "512-vs-2048 prefix CI width ratio in [0.80,1.25] and maximum endpoint difference <=0.02 AUROC",
                "wrong_formula_control": "deliberately incorrect t interval over bootstrap draws has width ratio <0.65 at 2048 vs 512 draws"},
            "truth": "analytic normal-score AUROC, not estimated by another Monte Carlo sample",
            "familywise_claim": False, "stopping": "fixed repetitions and draws; no extension or tuning after results",
            "limits": ["Five marginal coverage diagnostics, no joint 95% guarantee or universal coverage proof.",
                       "Independent simulated victim identities; shared training-data or trigger-family dependence is outside this model.",
                       "Four detector seeds in crossed simulation deliberately expose finite-seed limitations.",
                       "Historical score tables and scientific rankings are not re-analysed."]}


def _summarize_coverage(rows, name):
    covered = sum(row["covered"] for row in rows)
    interval = wilson_interval(covered, len(rows))
    coverage = covered / len(rows)
    return {"condition": name, "repetitions": len(rows), "covered": covered, "coverage": coverage,
            "coverage_ci95": interval, "mean_interval_width": float(np.mean([r["ci95"][1] - r["ci95"][0] for r in rows])),
            "mean_estimate": float(np.mean([r["estimate"] for r in rows])), "truth": rows[0]["truth"],
            "mean_bias": float(np.mean([r["estimate"] - r["truth"] for r in rows])),
            "criterion_met": bool(coverage >= 0.90 and interval[0] <= 0.95 <= interval[1])}


def run(output_dir: Path) -> dict:
    """Full fixed protocol: root freezes before invoking this function."""
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_spec()
    all_rows, summaries = [], []
    config = protocol["population_simulations"]
    conditions = [("fixed_detector_null", 1, 0.7, 0.0, "new_victims_fixed_detectors"),
                  ("fixed_detector_effect", 1, 1.0, 0.0, "new_victims_fixed_detectors"),
                  ("crossed_null", 4, 0.7, 0.3, "new_victims_new_detectors"),
                  ("crossed_effect", 4, 1.0, 0.3, "new_victims_new_detectors")]
    for condition_index, (name, detector_count, mu, noise, mode) in enumerate(conditions):
        rows = []
        for repeat in range(config["repetitions_per_condition"]):
            seed = config["seed_base"] + condition_index * 10000 + repeat
            scores, labels, truth, _ = _normal_scores(np.random.default_rng(seed), config["models_per_class"], detector_count, 0.7, mu, noise)
            estimate = paired_auroc(scores, labels)["effect"]
            boot = bootstrap_effects(scores, labels, 399, seed + 500000, mode)
            interval = percentile_interval(boot)
            rows.append({"condition": name, "repeat": repeat, "data_seed": seed, "bootstrap_seed": seed + 500000,
                         "estimate": estimate, "truth": truth, "ci95": interval, "covered": bool(interval[0] <= truth <= interval[1])})
        summaries.append(_summarize_coverage(rows, name))
        all_rows.extend(rows)
    fixed = protocol["fixed_victim_simulation"]
    _, labels, _, latent = _normal_scores(np.random.default_rng(fixed["latent_seed"]), fixed["models_per_class"], 1, 0.7, 1.0, 0.0)
    sigma = fixed["detector_noise_sd"]
    pair_delta = latent[labels == 1, :][:, None, :] - latent[labels == 0, :][None, :, :]
    auc_truth = ndtr(pair_delta / (np.sqrt(2) * sigma)).mean(axis=(0, 1))
    conditional_truth = float(auc_truth[1] - auc_truth[0])
    rows = []
    for repeat in range(fixed["repetitions"]):
        seed = fixed["draw_seed_base"] + repeat
        rng = np.random.default_rng(seed)
        scores = latent[None, :, :] + sigma * rng.normal(size=(fixed["detector_seeds"], len(labels), 2))
        estimate = paired_auroc(scores, labels)["effect"]
        boot = bootstrap_effects(scores, labels, 399, seed + 500000, "fixed_victims_new_detectors")
        interval = percentile_interval(boot)
        rows.append({"condition": "fixed_victims_new_detectors", "repeat": repeat, "data_seed": seed,
                     "bootstrap_seed": seed + 500000, "estimate": estimate, "truth": conditional_truth,
                     "ci95": interval, "covered": bool(interval[0] <= conditional_truth <= interval[1])})
    summaries.append(_summarize_coverage(rows, "fixed_victims_new_detectors"))
    all_rows.extend(rows)
    # Freeze one score table, then take increasing prefixes of identical draws.
    scores, labels, truth, _ = _normal_scores(np.random.default_rng(2026410500), 80, 1, 0.7, 1.0, 0.0)
    boot = bootstrap_effects(scores, labels, 2048, 2026410501, "new_victims_fixed_detectors")
    stability_rows = []
    for count in (128, 512, 2048):
        sample = boot[:count]
        interval = percentile_interval(sample)
        # Deliberate negative control: this treats Monte Carlo draws as units.
        margin = float(t.ppf(0.975, count - 1) * sample.std(ddof=1) / np.sqrt(count))
        wrong = [float(sample.mean() - margin), float(sample.mean() + margin)]
        stability_rows.append({"draws": count, "ci95": interval, "width": interval[1] - interval[0],
                               "incorrect_t_over_draws": wrong, "incorrect_width": 2 * margin})
    middle, large = stability_rows[1:]
    ratio = middle["width"] / large["width"]
    endpoint_difference = max(abs(a - b) for a, b in zip(middle["ci95"], large["ci95"]))
    wrong_ratio = large["incorrect_width"] / middle["incorrect_width"]
    stability = {"table_seed": 2026410500, "bootstrap_seed": 2026410501, "point": paired_auroc(scores, labels),
                 "analytic_truth": truth, "prefixes": stability_rows, "width_ratio_512_over_2048": ratio,
                 "max_endpoint_difference": endpoint_difference, "criterion_met": bool(0.8 <= ratio <= 1.25 and endpoint_difference <= 0.02),
                 "wrong_formula_width_ratio_2048_over_512": wrong_ratio, "negative_control_detected": bool(wrong_ratio < 0.65)}
    np.savez_compressed(output_dir / "uncertainty-stability.npz", scores=scores, labels=labels, bootstrap_effects=boot)
    (output_dir / "uncertainty-simulation-records.json").write_text(json.dumps(all_rows, indent=2, allow_nan=False) + "\n")
    success = all(row["criterion_met"] for row in summaries) and stability["criterion_met"] and stability["negative_control_detected"]
    result = {"protocol": protocol, "result": "met" if success else "not_met", "coverage": summaries,
              "stability": stability, "records_file": "uncertainty-simulation-records.json", "stability_file": "uncertainty-stability.npz",
              "limits": protocol["limits"] + ["A failed calibration criterion triggers a new version and fresh simulation seeds, not retrospective threshold changes."]}
    (output_dir / "uncertainty-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
