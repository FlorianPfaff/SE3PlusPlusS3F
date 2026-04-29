import csv
import json

import numpy as np
import numpy.testing as npt

from se3plusplus_s3f.s1r2.baseline_comparison import (
    BaselineComparisonConfig,
    run_baseline_comparison,
    run_baseline_comparison_on_trials,
    write_baseline_comparison_outputs,
)
from se3plusplus_s3f.s1r2.relaxed_s3f_pilot import PilotConfig, generate_pilot_trials


BEHAVIOR_METRICS = [
    "position_rmse",
    "orientation_mode_error_rad",
    "orientation_mean_error_rad",
    "mean_nees",
    "coverage_95",
]


def test_baseline_comparison_smoke_outputs_metrics(tmp_path):
    config = BaselineComparisonConfig(
        pilot=PilotConfig(grid_sizes=(8,), n_trials=1, n_steps=2),
        particle_count=64,
    )

    rows = run_baseline_comparison(config)

    assert len(rows) == 5
    assert {(row["filter"], row["variant"]) for row in rows} == {
        ("s3f", "baseline"),
        ("s3f", "r1"),
        ("s3f", "r1_r2"),
        ("ekf", "single_gaussian"),
        ("particle_filter", "bootstrap"),
    }
    assert all(np.isfinite(float(row["position_rmse"])) for row in rows)
    assert all(float(row["runtime_ms_per_step"]) > 0.0 for row in rows)

    outputs = write_baseline_comparison_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "note", "metadata"):
        assert outputs[output_name].is_file()

    with outputs["metrics"].open(newline="", encoding="utf-8") as file:
        written_rows = list(csv.DictReader(file))
    assert len(written_rows) == 5

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "baseline_comparison"
    assert metadata["metrics_rows"] == 5


def test_baseline_comparison_uses_reproducible_shared_trials():
    config = BaselineComparisonConfig(
        pilot=PilotConfig(grid_sizes=(8,), n_trials=1, n_steps=2),
        particle_count=64,
    )
    trials = generate_pilot_trials(config.pilot)

    rows_from_runner = _index_rows(run_baseline_comparison(config))
    rows_from_trials = _index_rows(run_baseline_comparison_on_trials(config, trials))

    assert set(rows_from_runner) == set(rows_from_trials)
    for key, expected_row in rows_from_runner.items():
        actual_row = rows_from_trials[key]
        for metric in BEHAVIOR_METRICS:
            npt.assert_allclose(
                float(actual_row[metric]),
                float(expected_row[metric]),
                rtol=1e-12,
                atol=1e-12,
            )


def _index_rows(rows):
    return {
        (str(row["filter"]), str(row["variant"]), str(row["grid_size"])): row
        for row in rows
    }
