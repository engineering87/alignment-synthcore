# Evaluation calibration v1

Six frozen calibration modules, with existing scientific code/results preserved.
See results/evaluation-calibration-v1/REPORT.md for findings and limitations.

Run from the repository root with NumPy, SciPy, PyTorch and pytest installed:

    python -m pytest -q tests/test_calibration_evaluator.py tests/test_calibration_terminal.py tests/test_calibration_measurements.py tests/test_calibration_replay.py
    python -m experiments.evaluation_calibration_v1.runner verify --output results/evaluation-calibration-v1

The saved protocol specifies all module designs. Freeze before new execution; run
into an unused directory containing the frozen protocol and source manifest.
Never overwrite existing run directories or reinterpret failed gates as passes.
The candidate evaluator is a separate API; historical runner integration is pending.
