"""Evidence summary report for S3+ x R3 relaxed S3F runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..s1r2.plotting import format_plot_list, save_figure
from .highres_reference import S3R3HighResReferenceConfig, run_s3r3_highres_reference_benchmark
from .relaxed_s3f_prototype import S3R3PrototypeConfig, VARIANT_LABELS, run_s3r3_relaxed_prototype, validate_s3r3_prototype_config


S3R3_EVIDENCE_VARIANTS = ("baseline", "r1", "r1_r2")

S3R3_EVIDENCE_CLAIM_FIELDNAMES = [
    "claim_id",
    "evidence_source",
    "comparison",
    "grid_size",
    "candidate_variant",
    "comparator_variant",
    "candidate_position_rmse",
    "comparator_position_rmse",
    "position_rmse_ratio",
    "position_rmse_gain_pct",
    "candidate_mean_nees",
    "comparator_mean_nees",
    "mean_nees_ratio",
    "candidate_coverage_95",
    "comparator_coverage_95",
    "coverage_delta",
    "candidate_runtime_ms_per_step",
    "comparator_runtime_ms_per_step",
    "runtime_ratio",
    "candidate_position_rmse_to_reference",
    "comparator_position_rmse_to_reference",
    "reference_rmse_ratio",
    "reference_rmse_gain_pct",
    "supports_accuracy_claim",
    "supports_consistency_claim",
    "supports_reference_claim",
    "supports_overall_claim",
]


@dataclass(frozen=True)
class S3R3EvidenceSummaryResult:
    """Container for the S3R3 evidence summary tables."""

    relaxed_metrics: list[dict[str, float | int | str]]
    highres_metrics: list[dict[str, float | int | str]]
    claims: list[dict[str, float | int | str | bool]]


@dataclass(frozen=True)
class S3R3EvidenceSummaryConfig:
    """Configuration for the combined S3R3 evidence summary."""

    prototype: S3R3PrototypeConfig = field(
        default_factory=lambda: S3R3PrototypeConfig(
            grid_sizes=(8, 16, 32),
            variants=S3R3_EVIDENCE_VARIANTS,
            n_trials=8,
            n_steps=8,
            seed=29,
        )
    )
    reference_grid_size: int = 64


def s3r3_evidence_summary_config_to_dict(config: S3R3EvidenceSummaryConfig) -> dict[str, Any]:
    """Return a JSON-serializable evidence summary config."""

    return json.loads(json.dumps(asdict(config)))


def run_s3r3_evidence_summary(
    config: S3R3EvidenceSummaryConfig = S3R3EvidenceSummaryConfig(),
) -> S3R3EvidenceSummaryResult:
    """Run the relaxed and high-resolution S3R3 reports and build claim rows."""

    validate_s3r3_prototype_config(
        config.prototype,
        reference_grid_size=config.reference_grid_size,
        required_variants=S3R3_EVIDENCE_VARIANTS,
    )
    relaxed_metrics = run_s3r3_relaxed_prototype(config.prototype)
    highres_metrics = run_s3r3_highres_reference_benchmark(
        S3R3HighResReferenceConfig(
            prototype=config.prototype,
            reference_grid_size=config.reference_grid_size,
        )
    )
    return S3R3EvidenceSummaryResult(
        relaxed_metrics=relaxed_metrics,
        highres_metrics=highres_metrics,
        claims=_build_claim_rows(relaxed_metrics, highres_metrics),
    )


def write_s3r3_evidence_summary_outputs(
    output_dir: Path,
    config: S3R3EvidenceSummaryConfig = S3R3EvidenceSummaryConfig(),
    write_plots: bool = True,
) -> dict[str, Path]:
    """Run the evidence summary and write metrics, claims, optional plots, metadata, and note."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_s3r3_evidence_summary(config)

    relaxed_path = output_dir / "s3r3_evidence_relaxed_metrics.csv"
    _write_csv(relaxed_path, result.relaxed_metrics)

    highres_path = output_dir / "s3r3_evidence_highres_metrics.csv"
    _write_csv(highres_path, result.highres_metrics)

    claims_path = output_dir / "s3r3_evidence_claims.csv"
    _write_csv(claims_path, result.claims, fieldnames=S3R3_EVIDENCE_CLAIM_FIELDNAMES)

    outputs = {"relaxed_metrics": relaxed_path, "highres_metrics": highres_path, "claims": claims_path}
    plot_paths = _write_plots(output_dir, result.claims) if write_plots else []
    outputs.update({plot_path.stem: plot_path for plot_path in plot_paths})

    note_path = output_dir / "s3r3_evidence_summary_note.md"
    _write_note(note_path, result, relaxed_path, highres_path, claims_path, plot_paths, config)
    outputs["note"] = note_path

    metadata_path = output_dir / "run_metadata.json"
    _write_metadata(metadata_path, result, config)
    outputs["metadata"] = metadata_path
    return outputs


def _build_claim_rows(
    relaxed_metrics: list[dict[str, float | int | str]],
    highres_metrics: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str | bool]]:
    claims: list[dict[str, float | int | str | bool]] = []
    relaxed_index = _index_by_grid_variant(relaxed_metrics)
    highres_index = _index_by_grid_variant(highres_metrics)
    for grid_size in _available_grid_sizes(relaxed_metrics, highres_metrics):
        claims.append(
            _claim_row(
                claim_id=f"truth_r1_r2_vs_baseline_{grid_size}",
                evidence_source="relaxed_truth",
                comparison="R1+R2 vs baseline on direct truth metrics",
                candidate=relaxed_index[(grid_size, "r1_r2")],
                comparator=relaxed_index[(grid_size, "baseline")],
            )
        )
        claims.append(
            _claim_row(
                claim_id=f"reference_r1_r2_vs_baseline_{grid_size}",
                evidence_source="highres_reference",
                comparison="R1+R2 vs baseline relative to dense S3F reference",
                candidate=highres_index[(grid_size, "r1_r2")],
                comparator=highres_index[(grid_size, "baseline")],
            )
        )
        claims.append(
            _claim_row(
                claim_id=f"inflation_r1_r2_vs_r1_{grid_size}",
                evidence_source="highres_reference",
                comparison="R1+R2 vs R1 to isolate covariance inflation",
                candidate=highres_index[(grid_size, "r1_r2")],
                comparator=highres_index[(grid_size, "r1")],
            )
        )
    return claims


def _index_by_grid_variant(rows: list[dict[str, float | int | str]]) -> dict[tuple[int, str], dict[str, float | int | str]]:
    return {(int(row["grid_size"]), str(row["variant"])): row for row in rows}


def _available_grid_sizes(
    relaxed_metrics: list[dict[str, float | int | str]],
    highres_metrics: list[dict[str, float | int | str]],
) -> list[int]:
    relaxed_grids = {int(row["grid_size"]) for row in relaxed_metrics}
    highres_grids = {int(row["grid_size"]) for row in highres_metrics}
    return sorted(relaxed_grids & highres_grids)


def _claim_row(
    claim_id: str,
    evidence_source: str,
    comparison: str,
    candidate: dict[str, float | int | str],
    comparator: dict[str, float | int | str],
) -> dict[str, float | int | str | bool]:
    candidate_rmse = _metric_value(candidate, "position_rmse", "position_rmse_to_truth")
    comparator_rmse = _metric_value(comparator, "position_rmse", "position_rmse_to_truth")
    candidate_nees = _metric_value(candidate, "mean_nees", "mean_nees_to_truth")
    comparator_nees = _metric_value(comparator, "mean_nees", "mean_nees_to_truth")
    candidate_coverage = _metric_value(candidate, "coverage_95", "coverage_95_to_truth")
    comparator_coverage = _metric_value(comparator, "coverage_95", "coverage_95_to_truth")
    candidate_runtime = _metric_value(candidate, "runtime_ms_per_step")
    comparator_runtime = _metric_value(comparator, "runtime_ms_per_step")
    candidate_reference = _optional_metric(candidate, "position_rmse_to_reference")
    comparator_reference = _optional_metric(comparator, "position_rmse_to_reference")

    rmse_ratio = _ratio(candidate_rmse, comparator_rmse)
    nees_ratio = _ratio(candidate_nees, comparator_nees)
    coverage_delta = candidate_coverage - comparator_coverage
    reference_ratio = _optional_ratio(candidate_reference, comparator_reference)
    supports_reference = reference_ratio != "" and float(reference_ratio) < 1.0
    supports_accuracy = rmse_ratio < 1.0
    supports_consistency = nees_ratio < 1.0 and coverage_delta >= 0.0
    return {
        "claim_id": claim_id,
        "evidence_source": evidence_source,
        "comparison": comparison,
        "grid_size": int(candidate["grid_size"]),
        "candidate_variant": candidate["variant"],
        "comparator_variant": comparator["variant"],
        "candidate_position_rmse": candidate_rmse,
        "comparator_position_rmse": comparator_rmse,
        "position_rmse_ratio": rmse_ratio,
        "position_rmse_gain_pct": _gain_percent(candidate_rmse, comparator_rmse),
        "candidate_mean_nees": candidate_nees,
        "comparator_mean_nees": comparator_nees,
        "mean_nees_ratio": nees_ratio,
        "candidate_coverage_95": candidate_coverage,
        "comparator_coverage_95": comparator_coverage,
        "coverage_delta": coverage_delta,
        "candidate_runtime_ms_per_step": candidate_runtime,
        "comparator_runtime_ms_per_step": comparator_runtime,
        "runtime_ratio": _ratio(candidate_runtime, comparator_runtime),
        "candidate_position_rmse_to_reference": candidate_reference,
        "comparator_position_rmse_to_reference": comparator_reference,
        "reference_rmse_ratio": reference_ratio,
        "reference_rmse_gain_pct": "" if reference_ratio == "" else _gain_percent(float(candidate_reference), float(comparator_reference)),
        "supports_accuracy_claim": supports_accuracy,
        "supports_consistency_claim": supports_consistency,
        "supports_reference_claim": supports_reference,
        "supports_overall_claim": supports_accuracy and supports_consistency and (evidence_source != "highres_reference" or supports_reference),
    }


def _metric_value(row: dict[str, float | int | str], *names: str) -> float:
    for name in names:
        if name in row and row[name] != "":
            return float(row[name])
    raise KeyError(names)


def _optional_metric(row: dict[str, float | int | str], name: str) -> float | str:
    return float(row[name]) if name in row and row[name] != "" else ""


def _ratio(candidate: float, comparator: float) -> float:
    return candidate / comparator if comparator else float("inf")


def _optional_ratio(candidate: float | str, comparator: float | str) -> float | str:
    return "" if candidate == "" or comparator == "" else _ratio(float(candidate), float(comparator))


def _gain_percent(candidate: float, comparator: float) -> float:
    return 100.0 * (1.0 - _ratio(candidate, comparator))


def _write_csv(
    path: Path,
    rows: list[dict[str, float | int | str | bool]],
    fieldnames: list[str] | None = None,
) -> None:
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[name] for name in columns])


def _write_metadata(path: Path, result: S3R3EvidenceSummaryResult, config: S3R3EvidenceSummaryConfig) -> None:
    metadata = {
        "config": s3r3_evidence_summary_config_to_dict(config),
        "experiment": "s3r3_evidence_summary",
        "relaxed_metrics_rows": len(result.relaxed_metrics),
        "highres_metrics_rows": len(result.highres_metrics),
        "claims_rows": len(result.claims),
        "claims_schema": S3R3_EVIDENCE_CLAIM_FIELDNAMES,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_note(
    path: Path,
    result: S3R3EvidenceSummaryResult,
    relaxed_path: Path,
    highres_path: Path,
    claims_path: Path,
    plot_paths: list[Path],
    config: S3R3EvidenceSummaryConfig,
) -> None:
    supported = [claim for claim in result.claims if bool(claim["supports_overall_claim"])]
    best_reference = min(result.highres_metrics, key=lambda row: float(row["position_rmse_to_reference"]))
    best_truth = min(result.relaxed_metrics, key=lambda row: float(row["position_rmse"]))
    lines = [
        "# S3+ x R3 Evidence Summary",
        "",
        "This report combines direct S3R3 relaxed metrics with a dense-grid S3F reference comparison.",
        "",
        f"Trials: {config.prototype.n_trials}",
        f"Steps per trial: {config.prototype.n_steps}",
        f"Grid sizes: {list(config.prototype.grid_sizes)}",
        f"Reference grid size: {config.reference_grid_size}",
        f"Cell sample count: {config.prototype.cell_sample_count}",
        f"Relaxed metrics: `{relaxed_path.name}`",
        f"High-resolution metrics: `{highres_path.name}`",
        f"Claims: `{claims_path.name}`",
        "",
        "## Best Rows",
        "",
        (
            f"Best direct truth RMSE: `{VARIANT_LABELS[str(best_truth['variant'])]}` at `{best_truth['grid_size']}` cells "
            f"with RMSE `{float(best_truth['position_rmse']):.4f}` and NEES `{float(best_truth['mean_nees']):.3f}`."
        ),
        (
            f"Best dense-reference match: `{VARIANT_LABELS[str(best_reference['variant'])]}` at `{best_reference['grid_size']}` cells "
            f"with RMSE-to-reference `{float(best_reference['position_rmse_to_reference']):.4f}`."
        ),
        "",
        "## Claims",
        "",
        _format_claims_table(result.claims),
        "",
        "## Interpretation",
        "",
        _interpret_claims(result.claims, supported),
        "",
        "Plots:",
        format_plot_list(plot_paths),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_claims_table(claims: list[dict[str, float | int | str | bool]]) -> str:
    header = "| Source | Grid | Candidate | Comparator | RMSE gain % | Ref gain % | NEES ratio | Coverage delta | Overall |"
    separator = "|---|---:|---|---|---:|---:|---:|---:|---|"
    rows = []
    for claim in claims:
        ref_gain = claim["reference_rmse_gain_pct"]
        rows.append(
            "| "
            f"{claim['evidence_source']} | "
            f"{int(claim['grid_size'])} | "
            f"{claim['candidate_variant']} | "
            f"{claim['comparator_variant']} | "
            f"{float(claim['position_rmse_gain_pct']):.1f} | "
            f"{_format_optional_float(ref_gain, '.1f')} | "
            f"{float(claim['mean_nees_ratio']):.3f} | "
            f"{float(claim['coverage_delta']):.3f} | "
            f"{'yes' if bool(claim['supports_overall_claim']) else 'no'} |"
        )
    return "\n".join([header, separator, *rows])


def _format_optional_float(value: float | int | str | bool, fmt: str) -> str:
    return "" if value == "" else format(float(value), fmt)


def _interpret_claims(
    claims: list[dict[str, float | int | str | bool]],
    supported: list[dict[str, float | int | str | bool]],
) -> str:
    if not supported:
        return "No claim row satisfies the configured accuracy and consistency checks."

    same_grid = [
        claim
        for claim in supported
        if str(claim["claim_id"]).startswith("reference_r1_r2_vs_baseline")
    ]
    inflation = [
        claim
        for claim in supported
        if str(claim["claim_id"]).startswith("inflation_r1_r2_vs_r1")
    ]
    direct = [
        claim
        for claim in supported
        if str(claim["claim_id"]).startswith("truth_r1_r2_vs_baseline")
    ]
    return " ".join(
        [
            _grid_summary("Direct truth R1+R2-vs-baseline support", direct),
            _grid_summary("Dense-reference R1+R2-vs-baseline support", same_grid),
            _grid_summary("Covariance-inflation support over R1 alone", inflation),
            f"Supported claim rows: `{len(supported)}/{len(claims)}`.",
        ]
    )


def _grid_summary(label: str, claims: list[dict[str, float | int | str | bool]]) -> str:
    if not claims:
        return f"{label}: none."
    grid_list = ", ".join(str(int(claim["grid_size"])) for claim in claims)
    return f"{label}: grids `{grid_list}`."


def _write_plots(output_dir: Path, claims: list[dict[str, float | int | str | bool]]) -> list[Path]:
    return [
        _write_gain_plot(output_dir, claims),
        _write_consistency_plot(output_dir, claims),
    ]


def _write_gain_plot(output_dir: Path, claims: list[dict[str, float | int | str | bool]]) -> Path:
    labels = [_claim_plot_label(claim) for claim in claims]
    y_positions = list(range(len(labels)))
    rmse_gain = [float(claim["position_rmse_gain_pct"]) for claim in claims]
    ref_gain = [0.0 if claim["reference_rmse_gain_pct"] == "" else float(claim["reference_rmse_gain_pct"]) for claim in claims]

    fig, ax = plt.subplots(figsize=(8.4, max(4.2, 0.34 * len(labels))))
    ax.barh([value - 0.18 for value in y_positions], rmse_gain, height=0.34, label="truth RMSE")
    ax.barh([value + 0.18 for value in y_positions], ref_gain, height=0.34, label="reference RMSE")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("RMSE gain over comparator [%]")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    return save_figure(fig, output_dir, "s3r3_evidence_rmse_gains.png")


def _write_consistency_plot(output_dir: Path, claims: list[dict[str, float | int | str | bool]]) -> Path:
    labels = [_claim_plot_label(claim) for claim in claims]
    fig, ax = plt.subplots(figsize=(8.4, max(4.2, 0.34 * len(labels))))
    ax.barh(range(len(labels)), [float(claim["mean_nees_ratio"]) for claim in claims], color="#4C78A8", alpha=0.88)
    ax.axvline(1.0, color="black", linewidth=1.0)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("NEES ratio, candidate / comparator")
    ax.grid(True, axis="x", alpha=0.3)
    return save_figure(fig, output_dir, "s3r3_evidence_nees_ratios.png")


def _claim_plot_label(claim: dict[str, float | int | str | bool]) -> str:
    source = "ref" if claim["evidence_source"] == "highres_reference" else "truth"
    return f"{source} {claim['candidate_variant']} vs {claim['comparator_variant']} ({claim['grid_size']})"
