import csv
import json

import numpy as np
import pytest

from se3plusplus_s3f.s3r3.evidence_summary import (
    S3R3EvidenceSummaryConfig,
    run_s3r3_evidence_summary,
    write_s3r3_evidence_summary_outputs,
)
from se3plusplus_s3f.s3r3.relaxed_s3f_prototype import S3R3PrototypeConfig


def test_s3r3_evidence_summary_smoke_outputs_claims(tmp_path):
    config = S3R3EvidenceSummaryConfig(
        prototype=S3R3PrototypeConfig(
            grid_sizes=(8,),
            n_trials=1,
            n_steps=2,
        ),
        reference_grid_size=16,
    )

    result = run_s3r3_evidence_summary(config)

    assert len(result.relaxed_metrics) == 3
    assert len(result.highres_metrics) == 3
    assert len(result.claims) == 3
    assert {claim["evidence_source"] for claim in result.claims} == {"relaxed_truth", "highres_reference"}
    assert {claim["claim_id"] for claim in result.claims} == {
        "inflation_r1_r2_vs_r1_8",
        "reference_r1_r2_vs_baseline_8",
        "truth_r1_r2_vs_baseline_8",
    }
    assert all(np.isfinite(float(claim["position_rmse_ratio"])) for claim in result.claims)
    assert all(np.isfinite(float(claim["mean_nees_ratio"])) for claim in result.claims)

    outputs = write_s3r3_evidence_summary_outputs(tmp_path / "out", config, write_plots=False)
    for output_name in ("relaxed_metrics", "highres_metrics", "claims", "note", "metadata"):
        assert outputs[output_name].is_file()

    with outputs["claims"].open(newline="", encoding="utf-8") as claims_file:
        assert len(list(csv.DictReader(claims_file))) == len(result.claims)

    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert metadata["experiment"] == "s3r3_evidence_summary"
    assert metadata["relaxed_metrics_rows"] == 3
    assert metadata["highres_metrics_rows"] == 3
    assert metadata["claims_rows"] == 3

    note_text = outputs["note"].read_text(encoding="utf-8")
    assert "## Claims" in note_text
    assert "Covariance-inflation support" in note_text


def test_s3r3_evidence_summary_requires_finer_reference():
    config = S3R3EvidenceSummaryConfig(
        prototype=S3R3PrototypeConfig(grid_sizes=(8, 16)),
        reference_grid_size=16,
    )

    with pytest.raises(ValueError, match="reference_grid_size"):
        run_s3r3_evidence_summary(config)
