import csv
import json

import numpy as np
import pytest

from se3plusplus_s3f.s3r3.particle_comparison import (
    S3R3ParticleComparisonConfig,
    run_s3r3_particle_comparison,
    write_s3r3_particle_comparison_outputs,
)
from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig


def test_s3r3_particle_comparison_smoke_outputs_summary(tmp_path):
    config = S3R3ParticleComparisonConfig(
        prototype=S3R3PrototypeConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
        ),
        prior_kappas=(2.0,),
        body_increment_scales=(1.0,),
        particle_counts=(16, 32),
    )

    result = run_s3r3_particle_comparison(config)

    assert len(result.metrics) == 5
    assert len(result.comparisons) == 1
    assert {row["filter"] for row in result.metrics} == {"s3f", "particle_filter"}
    assert {row["variant"] for row in result.metrics if row["filter"] == "s3f"} == {"baseline", "r1", "r1_r2"}
    assert {int(row["particle_count"]) for row in result.metrics if row["filter"] == "particle_filter"} == {16, 32}
    assert all(np.isfinite(float(row["position_rmse"])) for row in result.metrics)
    assert all(np.isfinite(float(row["runtime_ms_per_step"])) for row in result.metrics)

    summary = result.comparisons[0]
    assert int(summary["best_s3f_grid_size"]) == 8
    assert int(summary["best_particle_count"]) in {16, 32}
    assert np.isfinite(float(summary["best_particle_rmse_ratio"]))
    assert np.isfinite(float(summary["nearest_particle_rmse_ratio"]))

    outputs = write_s3r3_particle_comparison_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("metrics", "summary", "note", "metadata"):
        assert outputs[output_name].is_file()

    assert _csv_row_count(outputs["metrics"]) == len(result.metrics)
    assert _csv_row_count(outputs["summary"]) == len(result.comparisons)

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "s3r3_particle_comparison"
    assert metadata["metrics_rows"] == 5
    assert metadata["summary_rows"] == 1

    note_text = outputs["note"].read_text(encoding="utf-8")
    assert "## Headline" in note_text
    assert "Scenario Summary" in note_text


def test_s3r3_particle_comparison_rejects_nonpositive_particle_counts():
    config = S3R3ParticleComparisonConfig(
        prototype=S3R3PrototypeConfig(grid_sizes=(8,)),
        particle_counts=(0,),
    )

    with pytest.raises(ValueError, match="particle_counts"):
        run_s3r3_particle_comparison(config)


def _csv_row_count(path):
    with path.open(newline="", encoding="utf-8") as file:
        return sum(1 for _row in csv.DictReader(file))
