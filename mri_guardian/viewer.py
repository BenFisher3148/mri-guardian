"""Side-by-side visual report for the MRI Guardian Layer."""

from __future__ import annotations

import json
from html import escape
from typing import Any

import numpy as np
from icsd.core.serialise import hash_canonical_json

from .harness import (
    MULTI_SERIES_CASE_KIND,
    _extract_arrays,
    build_case_verification_report,
    _round_float,
)
from .integration import build_guardian_integration_narrative
from .timing import build_timing_overlay_snapshot

_MAX_DIFFERENCE_ROWS = 64


def build_side_by_side_view_model(
    state: dict[str, Any],
    *,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic model for side-by-side AI versus verified review."""
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _build_multi_series_view_model(state)
    report = build_case_verification_report(state, profile_name=profile_name)
    if _is_minimal_structural_report(report):
        return _build_minimal_single_series_view_model(state, report)
    arrays, failure = _extract_arrays(state)
    if failure is not None:
        raise ValueError(failure.message)
    analysis = report["analysis"]
    manifest = report["manifest"]
    reasons = report["failure_reasons"]
    tiers = report["verification_tiers"]
    decision_summary = report["decision_summary"]
    ai_reconstruction = np.asarray(arrays["ai_reconstruction"], dtype=np.float64)
    ai_mask = np.asarray(analysis["ai_mask"], dtype=np.int64)
    verified_reconstruction = np.asarray(analysis["derived_reconstruction"], dtype=np.float64)
    verified_mask = np.asarray(analysis["verified_mask"], dtype=np.int64)
    intensity_mismatch_mask = (
        np.abs(ai_reconstruction - verified_reconstruction)
        > float(manifest["ai_reconstruction_match"]["max_abs_tolerance"])
    )
    rejected_mask = ((ai_mask == 1) & (verified_mask == 0)).astype(np.int64)
    missing_mask = ((ai_mask == 0) & (verified_mask == 1)).astype(np.int64)

    intensity_mismatch_count = int(np.count_nonzero(intensity_mismatch_mask))
    rejected_count = int(np.count_nonzero(rejected_mask))
    mask_delta = int(analysis["mask_voxel_delta"])
    intensity_min = _round_float(float(min(ai_reconstruction.min(), verified_reconstruction.min())))
    intensity_max = _round_float(float(max(ai_reconstruction.max(), verified_reconstruction.max())))

    slices: list[dict[str, Any]] = []
    for slice_index in range(ai_reconstruction.shape[0]):
        slice_mismatch_mask = intensity_mismatch_mask[slice_index]
        slice_rejected_mask = rejected_mask[slice_index]
        slice_missing_mask = missing_mask[slice_index]
        slices.append(
            {
                "slice_index": slice_index,
                "slice_label": f"Slice {slice_index + 1}",
                "ai_reconstruction": _array_to_nested_lists(ai_reconstruction[slice_index]),
                "verified_reconstruction": _array_to_nested_lists(
                    verified_reconstruction[slice_index]
                ),
                "ai_mask": ai_mask[slice_index].astype(np.int64).tolist(),
                "verified_mask": verified_mask[slice_index].astype(np.int64).tolist(),
                "intensity_mismatch_mask": slice_mismatch_mask.astype(np.int64).tolist(),
                "rejected_mask": slice_rejected_mask.astype(np.int64).tolist(),
                "missing_mask": slice_missing_mask.astype(np.int64).tolist(),
                "intensity_mismatch_count": int(np.count_nonzero(slice_mismatch_mask)),
                "rejected_count": int(np.count_nonzero(slice_rejected_mask)),
                "mask_delta": int(
                    np.count_nonzero(slice_rejected_mask) + np.count_nonzero(slice_missing_mask)
                ),
            }
        )

    difference_rows, total_difference_count = _difference_rows(
        ai_reconstruction=ai_reconstruction,
        verified_reconstruction=verified_reconstruction,
        ai_mask=ai_mask,
        verified_mask=verified_mask,
        intensity_mismatch_mask=intensity_mismatch_mask,
        rejected_mask=rejected_mask,
        missing_mask=missing_mask,
    )
    verification = _build_verification_summary(
        report=report,
        reasons=reasons,
        tiers=tiers,
        release_eligible=bool(report["release_eligible"]),
    )
    timing = build_timing_overlay_snapshot(manifest["profile_name"])
    findings = _build_findings(
        report=report,
        reasons=reasons,
        tiers=tiers,
        mask_delta=mask_delta,
        rejected_count=rejected_count,
        intensity_mismatch_count=intensity_mismatch_count,
        claimed_volume=_round_float(float(state["tumor_volume_mm3"])),
        computed_volume=analysis["computed_tumor_volume_mm3"],
        volume_verified=bool(manifest["tumor_volume_proof"]["verified"]),
    )
    integration = _build_integration_summary(
        release_eligible=bool(report["release_eligible"]),
        report=report,
        primary_reason=verification["primary_reason_summary"],
    )
    clinical_summary = _build_clinical_summary(
        state=state,
        analysis=analysis,
        verification=verification,
        release_eligible=bool(report["release_eligible"]),
        report=report,
    )
    return {
        "title": "MRI Guardian Layer",
        "subtitle": "Verification proxy between the reconstruction AI and the PACS/workstation boundary.",
        "profile_name": report["profile"]["profile_name"],
        "profile_revision": report["profile"]["profile_revision"],
        "profile_digest": report["profile"]["profile_digest"],
        "icsd_policy_id": report["profile"]["icsd_policy_id"],
        "icsd_policy_version": report["profile"]["icsd_policy_version"],
        "ai_label": "AI Reconstruction + Proposed Tumour Mask",
        "verified_label": "Guardian Layer Reconstruction + Verified Mask",
        "release_eligible": bool(report["release_eligible"]),
        "decision": report["decision"],
        "decision_summary": decision_summary,
        "verification": verification,
        "timing": timing,
        "findings": findings,
        "clinical_summary": clinical_summary,
        "integration": integration,
        "summary": {
            "mask_voxel_delta": mask_delta,
            "rejected_region_count": rejected_count,
            "intensity_mismatch_voxel_count": intensity_mismatch_count,
            "verified_mask_voxel_count": int(analysis["verified_mask_voxel_count"]),
            "claimed_tumor_volume_mm3": _round_float(float(state["tumor_volume_mm3"])),
            "verified_tumor_volume_mm3": analysis["computed_tumor_volume_mm3"],
            "volume_verified": manifest["tumor_volume_proof"]["verified"],
            "max_ai_abs_error": manifest["ai_reconstruction_match"]["max_abs_error"],
            "max_measurement_abs_error": manifest["measurement_consistency"]["max_abs_error"],
        },
        "intensity_window": {
            "min": intensity_min,
            "max": intensity_max,
        },
        "differences": {
            "rows": difference_rows,
            "omitted_count": max(0, total_difference_count - len(difference_rows)),
        },
        "slices": slices,
    }


def _build_multi_series_view_model(state: dict[str, Any]) -> dict[str, Any]:
    report = build_case_verification_report(state)
    primary_series = next(
        item for item in report["series_reports"] if item["resolved_role"] == "primary"
    )
    timing_profiles = [
        {
            "series_id": item["series_id"],
            "profile_name": item["profile"]["profile_name"],
            "timing": build_timing_overlay_snapshot(item["profile"]["profile_name"]),
        }
        for item in report["series_reports"]
    ]
    return {
        "case_kind": MULTI_SERIES_CASE_KIND,
        "title": "MRI Guardian Layer",
        "subtitle": "Study-level verification across multiple MRI series.",
        "release_eligible": bool(report["release_eligible"]),
        "decision": report["decision"],
        "decision_summary": report["decision_summary"],
        "verification": {
            "headline": _verification_headline(report),
            "primary_reason_summary": _viewer_primary_reason_summary(report),
            "reasons": (
                report["failure_reasons"]
                if report["failure_reasons"]
                else _decision_reason_items(report["decision"])
            ),
            "tiers": [
                {
                    "code": key,
                    "title": value["title"],
                    "status": value["status"],
                    "summary": value["summary"],
                }
                for key, value in report["verification_tiers"].items()
            ],
        },
        "study": {
            "required_series_ids": report["analysis"]["required_series_ids"],
            "supporting_series_ids": report["analysis"]["supporting_series_ids"],
            "auxiliary_series_ids": report["analysis"]["auxiliary_series_ids"],
            "released_series_ids": report["analysis"]["released_series_ids"],
            "withheld_series_ids": report["analysis"]["withheld_series_ids"],
            "geometry_verification": report["analysis"]["geometry_verification"],
            "lesion_corroboration": report["analysis"]["lesion_corroboration"],
            "profile_descriptor_hash": hash_canonical_json(
                report["contracts"]["verification_profile_descriptor"]
            ),
        },
        "series": [
            {
                "series_id": item["series_id"],
                "series_uid": item["series_uid"],
                "sequence_id": item["sequence_id"],
                "profile_name": item["profile"]["profile_name"],
                "resolved_role": item["resolved_role"],
                "gating_status": item["gating_status"],
                "series_proof_status": item["series_proof_status"],
                "series_release_eligible": item["series_release_eligible"],
                "downstream_release_allowed": item["downstream_release_allowed"],
                "primary_reason_summary": item["primary_reason_summary"],
                "verified_mask_voxel_count": item["analysis"]["verified_mask_voxel_count"],
                "computed_tumor_volume_mm3": item["analysis"]["computed_tumor_volume_mm3"],
                "grid_signature": item["grid_signature"],
            }
            for item in report["series_reports"]
        ],
        "timing_profiles": timing_profiles,
        "primary_profile_name": primary_series["profile"]["profile_name"],
    }


def _is_minimal_structural_report(report: dict[str, Any]) -> bool:
    return (
        report["decision"]["decision_class"] == "blocked_structural"
        and report["analysis"].get("derived_reconstruction_hash") is None
    )


def _build_minimal_single_series_view_model(
    state: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    verification = _build_verification_summary(
        report=report,
        reasons=report["failure_reasons"],
        tiers=report["verification_tiers"],
        release_eligible=False,
    )
    case_id = state.get("case_id")
    timing = build_timing_overlay_snapshot(report["profile"]["profile_name"])
    claimed_volume = state.get("tumor_volume_mm3")
    claimed_volume_display = (
        _round_float(float(claimed_volume))
        if isinstance(claimed_volume, (int, float))
        else "Unavailable"
    )
    return {
        "title": "MRI Guardian Layer",
        "subtitle": "Verification proxy between the reconstruction AI and the PACS/workstation boundary.",
        "profile_name": report["profile"]["profile_name"],
        "profile_revision": report["profile"]["profile_revision"],
        "profile_digest": report["profile"]["profile_digest"],
        "icsd_policy_id": report["profile"]["icsd_policy_id"],
        "icsd_policy_version": report["profile"]["icsd_policy_version"],
        "ai_label": "AI Reconstruction + Proposed Tumour Mask",
        "verified_label": "Guardian Layer Reconstruction + Verified Mask",
        "release_eligible": False,
        "decision": report["decision"],
        "decision_summary": report["decision_summary"],
        "verification": verification,
        "timing": timing,
        "findings": [
            "Guardian structural intake blocked proof construction before physical consistency could be evaluated.",
            *[reason["message"] for reason in report["decision"]["decision_reasons"]],
        ],
        "clinical_summary": {
            "status_state": "blocked",
            "status_label": "NOT VERIFIED",
            "status_note": (
                "Blocked before downstream handoff. "
                f"Reason: {verification['primary_reason_summary']}."
            ),
            "tumor_volume_cm3": "Unavailable",
            "tumor_volume_note": (
                "Verified tumour volume was not computed because structural intake "
                "did not allow proof construction."
            ),
            "confidence_label": "Structural intake block",
            "confidence_note": (
                "The Guardian Layer refused downstream release before physical proof "
                "construction could begin."
            ),
            "audit_id": _short_audit_id(_minimal_structural_audit_hash(state, report)),
            "audit_note": (
                "Deterministic structural-block fingerprint for "
                f"{case_id if isinstance(case_id, str) and case_id.strip() else 'the received case'}."
            ),
        },
        "integration": _build_integration_summary(
            release_eligible=False,
            report=report,
            primary_reason=verification["primary_reason_summary"],
        ),
        "summary": {
            "mask_voxel_delta": 0,
            "rejected_region_count": 0,
            "intensity_mismatch_voxel_count": 0,
            "verified_mask_voxel_count": "Unavailable",
            "claimed_tumor_volume_mm3": claimed_volume_display,
            "verified_tumor_volume_mm3": "Unavailable",
            "volume_verified": False,
            "max_ai_abs_error": "Unavailable",
            "max_measurement_abs_error": "Unavailable",
        },
        "intensity_window": {"min": 0.0, "max": 1.0},
        "differences": {"rows": [], "omitted_count": 0},
        "slices": [],
    }


def _minimal_structural_audit_hash(
    state: dict[str, Any],
    report: dict[str, Any],
) -> str:
    return hash_canonical_json(
        {
            "case_id": state.get("case_id"),
            "study_uid": state.get("study_uid"),
            "series_uid": state.get("series_uid"),
            "input_state_hash": report["analysis"].get("input_state_hash"),
            "decision": report["decision"],
        }
    ).upper()


def _render_multi_series_report_html(
    state: dict[str, Any],
    *,
    report_title: str | None = None,
) -> str:
    view_model = _build_multi_series_view_model(state)
    title = report_title if report_title is not None else view_model["title"]
    series_rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(item['series_id'])}</td>"
            f"<td>{escape(item['resolved_role'])}</td>"
            f"<td>{escape(item['gating_status'])}</td>"
            f"<td>{escape(item['profile_name'])}</td>"
            f"<td>{escape(item['series_proof_status'])}</td>"
            f"<td>{'Yes' if item['downstream_release_allowed'] else 'No'}</td>"
            f"<td>{escape(item['primary_reason_summary'])}</td>"
            "</tr>"
        )
        for item in view_model["series"]
    )
    tier_markup = "\n".join(
        (
            "<li>"
            f"<strong>{escape(item['title'])}</strong>: "
            f"{escape(item['status'])} - {escape(item['summary'])}"
            "</li>"
        )
        for item in view_model["verification"]["tiers"]
    )
    reason_markup = "\n".join(
        f"<li><strong>{escape(item['code'])}</strong>: {escape(item['message'])}</li>"
        for item in view_model["verification"]["reasons"]
    )
    if not reason_markup:
        reason_markup = "<li>all_blocking_checks_passed</li>"
    released_series = ", ".join(view_model["study"]["released_series_ids"]) or "none"
    withheld_series = ", ".join(view_model["study"]["withheld_series_ids"]) or "none"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Aptos", "Segoe UI Variable", "Trebuchet MS", sans-serif;
      background: linear-gradient(135deg, #f8f1e6 0%, #ebe1d2 100%);
      color: #101820;
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 18px;
    }}
    section {{
      background: rgba(255, 248, 239, 0.94);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 18px 40px rgba(16, 24, 32, 0.12);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid rgba(16, 24, 32, 0.12);
      text-align: left;
      vertical-align: top;
    }}
    .status {{
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <p class="status">{escape(view_model['verification']['headline'])}</p>
      <h1>{escape(title)}</h1>
      <p>{escape(view_model['subtitle'])}</p>
      <p>Primary reason: {escape(view_model['verification']['primary_reason_summary'])}</p>
      <p>Released series: {escape(released_series)}</p>
      <p>Withheld series: {escape(withheld_series)}</p>
    </section>
    <section>
      <h2>Study Gate</h2>
      <ul>
        <li>Required series: {escape(", ".join(view_model['study']['required_series_ids']))}</li>
        <li>Supporting series: {escape(", ".join(view_model['study']['supporting_series_ids']))}</li>
        <li>Auxiliary series: {escape(", ".join(view_model['study']['auxiliary_series_ids']))}</li>
        <li>Geometry rule: {escape(view_model['study']['geometry_verification']['rule'])}</li>
        <li>Corroboration rule: {escape(view_model['study']['lesion_corroboration']['rule'])}</li>
      </ul>
    </section>
    <section>
      <h2>Verification Tiers</h2>
      <ul>{tier_markup}</ul>
    </section>
    <section>
      <h2>Series Register</h2>
      <table>
        <thead>
          <tr>
            <th>Series</th>
            <th>Role</th>
            <th>Gate</th>
            <th>Profile</th>
            <th>Proof</th>
            <th>Released</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {series_rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Failure Reasons</h2>
      <ul>{reason_markup}</ul>
    </section>
  </main>
</body>
</html>"""


def render_side_by_side_report_html(
    state: dict[str, Any],
    *,
    report_title: str | None = None,
    profile_name: str | None = None,
) -> str:
    """Render a self-contained HTML report for human review."""
    if state.get("case_kind") == MULTI_SERIES_CASE_KIND:
        return _render_multi_series_report_html(state, report_title=report_title)
    view_model = build_side_by_side_view_model(state, profile_name=profile_name)
    title = report_title if report_title is not None else view_model["title"]
    summary = view_model["summary"]
    verification = view_model["verification"]
    timing = view_model["timing"]
    benchmark_markup = _benchmark_list_markup(timing)
    streaming_markup = _streaming_strategy_markup(timing["streaming_strategy"])
    tier_cards_markup = _tier_cards_markup(verification["tiers"])
    findings_markup = "\n".join(
        f"<li>{escape(finding)}</li>" for finding in view_model["findings"]
    )
    differences_markup = _difference_table_markup(view_model["differences"])
    reason_markup = "\n".join(
        (
            "<li>"
            f"<strong>{escape(reason['code'])}</strong>: {escape(reason['message'])}"
            "</li>"
        )
        for reason in verification["reasons"]
    )
    data_json = json.dumps(view_model, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --paper: #f6f0e7;
      --ink: #101820;
      --signal-cool: #00d6ff;
      --signal-warm: #ffc24b;
      --signal-safe: #36e28f;
      --signal-danger: #ff5a5f;
      --panel-dark: #0f1721;
      --panel-mid: #162332;
      --card: rgba(255, 248, 239, 0.92);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Aptos", "Segoe UI Variable", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(0, 214, 255, 0.16), transparent 35%),
        radial-gradient(circle at top right, rgba(255, 194, 75, 0.24), transparent 30%),
        linear-gradient(135deg, #f8f1e6 0%, #ebe1d2 100%);
    }}

    .shell {{
      width: min(1280px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}

    .hero {{
      position: relative;
      overflow: hidden;
      border-radius: 28px;
      padding: 28px;
      color: #f5f7fb;
      background:
        linear-gradient(115deg, rgba(9, 22, 35, 0.98), rgba(22, 35, 50, 0.94)),
        linear-gradient(90deg, rgba(0, 214, 255, 0.12), rgba(255, 194, 75, 0.12));
      box-shadow: 0 24px 60px rgba(10, 15, 22, 0.24);
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -40px -60px auto;
      width: 220px;
      height: 220px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(255, 194, 75, 0.32), transparent 70%);
      filter: blur(10px);
    }}

    .eyebrow {{
      margin: 0 0 10px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-size: 0.78rem;
      color: rgba(255, 255, 255, 0.72);
    }}

    h1 {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2rem, 5vw, 3.4rem);
      line-height: 0.98;
      max-width: 12ch;
    }}

    .subtitle {{
      max-width: 56ch;
      margin: 14px 0 22px;
      color: rgba(245, 247, 251, 0.86);
      font-size: 1rem;
      line-height: 1.5;
    }}

    .status-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 12px;
    }}

    .clinical-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 16px 0 18px;
    }}

    .clinical-card {{
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}

    .clinical-label {{
      display: block;
      color: rgba(245, 247, 251, 0.68);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      margin-bottom: 10px;
    }}

    .clinical-value {{
      font-size: clamp(1rem, 2.4vw, 1.5rem);
      font-weight: 800;
      line-height: 1.1;
    }}

    .clinical-value-pass {{
      color: #ddffec;
    }}

    .clinical-value-fail {{
      color: #ffe5e6;
    }}

    .clinical-note {{
      margin: 8px 0 0;
      color: rgba(245, 247, 251, 0.78);
      font-size: 0.86rem;
      line-height: 1.45;
    }}

    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 700;
      letter-spacing: 0.02em;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.12);
    }}

    .status-pass {{
      color: #ddffec;
      background: rgba(54, 226, 143, 0.18);
    }}

    .status-fail {{
      color: #ffe5e6;
      background: rgba(255, 90, 95, 0.18);
    }}

    .failure-banner {{
      margin: 12px 0 20px;
      padding: 18px 20px;
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}

    .failure-banner h2 {{
      margin: 0;
      font-size: clamp(1.4rem, 3vw, 2.2rem);
      line-height: 1.05;
      letter-spacing: 0.02em;
      text-transform: none;
    }}

    .failure-reason {{
      margin: 10px 0 0;
      font-size: 1rem;
      color: rgba(245, 247, 251, 0.92);
    }}

    .failure-list {{
      margin: 12px 0 0;
      padding-left: 18px;
      color: rgba(245, 247, 251, 0.86);
      line-height: 1.45;
    }}

    .integration-map {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
      gap: 12px;
      align-items: stretch;
      margin: 18px 0 20px;
    }}

    .integration-node {{
      padding: 18px;
      border-radius: 22px;
      background: rgba(255, 248, 239, 0.94);
      border: 1px solid rgba(16, 24, 32, 0.08);
      box-shadow: 0 14px 30px rgba(15, 23, 33, 0.1);
    }}

    .integration-kicker {{
      display: inline-flex;
      font-size: 0.74rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #556573;
      margin-bottom: 10px;
    }}

    .integration-node h2 {{
      margin: 0 0 8px;
      font-size: 1.1rem;
      color: #101820;
    }}

    .integration-node p {{
      margin: 0;
      color: #334150;
      line-height: 1.55;
    }}

    .integration-arrow {{
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 42px;
      font-size: 1.3rem;
      color: #526170;
    }}

    .integration-story-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 0 0 18px;
    }}

    .integration-story-card {{
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(255, 248, 239, 0.94);
      border: 1px solid rgba(16, 24, 32, 0.08);
      box-shadow: 0 14px 30px rgba(15, 23, 33, 0.08);
    }}

    .integration-story-card h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
      color: #101820;
    }}

    .integration-story-card p {{
      margin: 0;
      color: #334150;
      line-height: 1.5;
      font-size: 0.92rem;
    }}

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
    }}

    .timing-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 18px 0 14px;
    }}

    .timing-card {{
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(7, 15, 24, 0.34);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}

    .timing-label {{
      display: block;
      color: rgba(255, 255, 255, 0.7);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 8px;
    }}

    .timing-value {{
      font-size: clamp(1.2rem, 2.8vw, 1.9rem);
      font-weight: 800;
      line-height: 1;
    }}

    .timing-meta {{
      margin: 10px 0 0;
      color: rgba(245, 247, 251, 0.72);
      font-size: 0.84rem;
      line-height: 1.45;
    }}

    .performance-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin: 18px 0 18px;
    }}

    .performance-card {{
      padding: 16px;
      border-radius: 20px;
      background: rgba(7, 15, 24, 0.34);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}

    .performance-card h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
      color: #f5f7fb;
    }}

    .performance-copy,
    .performance-note {{
      margin: 0 0 10px;
      color: rgba(245, 247, 251, 0.8);
      font-size: 0.9rem;
      line-height: 1.45;
    }}

    .performance-list {{
      margin: 0;
      padding-left: 18px;
      color: rgba(245, 247, 251, 0.84);
      font-size: 0.88rem;
      line-height: 1.45;
    }}

    .performance-list li + li {{
      margin-top: 8px;
    }}

    .tier-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 18px 0 18px;
    }}

    .tier-card {{
      padding: 16px;
      border-radius: 20px;
      background: rgba(7, 15, 24, 0.34);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}

    .tier-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 10px;
    }}

    .tier-title {{
      margin: 0;
      font-size: 1rem;
      line-height: 1.2;
    }}

    .tier-subtitle {{
      margin: 6px 0 0;
      color: rgba(245, 247, 251, 0.68);
      font-size: 0.82rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    .tier-status {{
      display: inline-flex;
      align-items: center;
      padding: 8px 10px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
    }}

    .tier-status-pass {{
      color: #ddffec;
      background: rgba(54, 226, 143, 0.18);
    }}

    .tier-status-fail {{
      color: #ffe5e6;
      background: rgba(255, 90, 95, 0.18);
    }}

    .tier-status-warn {{
      color: #fff0d1;
      background: rgba(255, 194, 75, 0.2);
    }}

    .tier-scope {{
      margin: 0 0 10px;
      color: rgba(245, 247, 251, 0.78);
      font-size: 0.86rem;
    }}

    .tier-summary {{
      margin: 0 0 12px;
      color: rgba(245, 247, 251, 0.9);
      font-size: 0.92rem;
      line-height: 1.45;
    }}

    .tier-checks {{
      margin: 0;
      padding-left: 18px;
      color: rgba(245, 247, 251, 0.82);
      font-size: 0.88rem;
      line-height: 1.4;
    }}

    .tier-checks li + li {{
      margin-top: 6px;
    }}

    .metric-card {{
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 248, 239, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.08);
      backdrop-filter: blur(10px);
    }}

    .metric-label {{
      display: block;
      margin-bottom: 8px;
      color: rgba(255, 255, 255, 0.68);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.09em;
    }}

    .metric-value {{
      font-size: clamp(1.4rem, 3vw, 2.2rem);
      font-weight: 800;
      line-height: 1;
    }}

    .metric-note {{
      margin-top: 8px;
      color: rgba(255, 255, 255, 0.72);
      font-size: 0.88rem;
      line-height: 1.35;
    }}

    .findings {{
      margin: 18px 0 0;
      padding-left: 18px;
      color: rgba(245, 247, 251, 0.9);
      line-height: 1.45;
    }}

    .findings li + li {{
      margin-top: 6px;
    }}

    .legend,
    .differences,
    .slice-grid {{
      margin-top: 24px;
    }}

    .legend,
    .differences {{
      background: var(--card);
      border: 1px solid rgba(16, 24, 32, 0.08);
      border-radius: 22px;
      padding: 20px 22px;
      box-shadow: 0 14px 30px rgba(44, 36, 29, 0.08);
    }}

    h2 {{
      margin: 0 0 14px;
      font-size: 1.05rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }}

    .legend-items {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 999px;
      background: rgba(16, 24, 32, 0.04);
    }}

    .swatch {{
      width: 18px;
      height: 18px;
      border-radius: 4px;
      border: 2px solid rgba(16, 24, 32, 0.16);
    }}

    .swatch-ai {{
      background: rgba(0, 214, 255, 0.45);
    }}

    .swatch-verified {{
      background: rgba(54, 226, 143, 0.45);
    }}

    .swatch-mismatch {{
      background: rgba(255, 194, 75, 0.45);
    }}

    .swatch-rejected {{
      background:
        repeating-linear-gradient(
          135deg,
          rgba(255, 90, 95, 0.82),
          rgba(255, 90, 95, 0.82) 4px,
          rgba(255, 90, 95, 0.14) 4px,
          rgba(255, 90, 95, 0.14) 8px
        );
    }}

    .slice-grid {{
      display: grid;
      gap: 18px;
    }}

    .slice-card {{
      overflow: hidden;
      border-radius: 24px;
      background: rgba(255, 250, 245, 0.96);
      border: 1px solid rgba(16, 24, 32, 0.08);
      box-shadow: 0 18px 36px rgba(33, 29, 24, 0.08);
    }}

    .slice-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 10px;
      padding: 18px 20px 12px;
      background: linear-gradient(180deg, rgba(16, 24, 32, 0.04), rgba(16, 24, 32, 0));
    }}

    .slice-title {{
      margin: 0;
      font-size: 1.3rem;
      font-weight: 800;
    }}

    .slice-stats {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 0.88rem;
    }}

    .slice-stat {{
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(16, 24, 32, 0.06);
    }}

    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      padding: 0 20px 22px;
    }}

    .panel {{
      padding: 14px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(15, 23, 33, 0.98), rgba(24, 36, 48, 0.98));
      color: #f4f6f8;
    }}

    .panel h3 {{
      margin: 0 0 8px;
      font-size: 0.95rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}

    .panel p {{
      margin: 0 0 12px;
      color: rgba(244, 246, 248, 0.76);
      font-size: 0.88rem;
    }}

    .panel canvas {{
      width: 100%;
      border-radius: 14px;
      background: #0b1017;
      image-rendering: pixelated;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}

    th,
    td {{
      padding: 10px 12px;
      text-align: left;
      border-top: 1px solid rgba(16, 24, 32, 0.08);
    }}

    th {{
      border-top: none;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(16, 24, 32, 0.62);
    }}

    .chip {{
      display: inline-flex;
      padding: 6px 8px;
      border-radius: 999px;
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .chip-rejected {{
      color: #7a1113;
      background: rgba(255, 90, 95, 0.16);
    }}

    .chip-missing {{
      color: #145f37;
      background: rgba(54, 226, 143, 0.18);
    }}

    .chip-mismatch {{
      color: #7b4f00;
      background: rgba(255, 194, 75, 0.22);
    }}

    .footnote {{
      margin-top: 12px;
      color: rgba(16, 24, 32, 0.65);
      font-size: 0.86rem;
      line-height: 1.45;
    }}

    @media (max-width: 860px) {{
      .integration-map {{
        grid-template-columns: minmax(0, 1fr);
      }}

      .integration-arrow {{
        min-height: 24px;
      }}

      .panel-grid {{
        grid-template-columns: 1fr;
      }}

      .shell {{
        width: min(100vw - 18px, 1280px);
      }}

      .hero {{
        padding: 22px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">ICSD Verification Layer</p>
      <h1>{escape(title)}</h1>
      <p class="subtitle">{escape(view_model["subtitle"])}</p>
      <div class="status-row">
        <span class="status-pill {'status-pass' if view_model['release_eligible'] else 'status-fail'}">
          {escape(verification["headline"])}
        </span>
        <span class="status-pill">Profile {escape(view_model["profile_name"])} r{escape(view_model["profile_revision"])}</span>
      </div>
      <div class="clinical-grid">
        <article class="clinical-card">
          <span class="clinical-label">Status</span>
          <div class="clinical-value {'clinical-value-pass' if view_model['clinical_summary']['status_state'] == 'verified' else 'clinical-value-fail'}">
            {escape(view_model["clinical_summary"]["status_label"])} {'&#x2705;' if view_model['clinical_summary']['status_state'] == 'verified' else '&#x274C;'}
          </div>
          <p class="clinical-note">{escape(view_model["clinical_summary"]["status_note"])}</p>
        </article>
        <article class="clinical-card">
          <span class="clinical-label">Tumour Volume</span>
          <div class="clinical-value">{escape(view_model["clinical_summary"]["tumor_volume_cm3"])} cm<sup>3</sup></div>
          <p class="clinical-note">{escape(view_model["clinical_summary"]["tumor_volume_note"])}</p>
        </article>
        <article class="clinical-card">
          <span class="clinical-label">Confidence</span>
          <div class="clinical-value">{escape(view_model["clinical_summary"]["confidence_label"])}</div>
          <p class="clinical-note">{escape(view_model["clinical_summary"]["confidence_note"])}</p>
        </article>
        <article class="clinical-card">
          <span class="clinical-label">Audit ID</span>
          <div class="clinical-value">{escape(view_model["clinical_summary"]["audit_id"])}</div>
          <p class="clinical-note">{escape(view_model["clinical_summary"]["audit_note"])}</p>
        </article>
      </div>
      <div class="timing-grid">
        <article class="timing-card">
          <span class="timing-label">Reconstruction</span>
          <div class="timing-value">{_display_ms(timing["reconstruction_ms"])}</div>
        </article>
        <article class="timing-card">
          <span class="timing-label">Verification</span>
          <div class="timing-value">{_display_ms(timing["verification_ms"])}</div>
        </article>
        <article class="timing-card">
          <span class="timing-label">Total Overhead</span>
          <div class="timing-value">+{_display_percent(timing["overhead_percent"])}</div>
        </article>
        <article class="timing-card">
          <span class="timing-label">Worst-Case Bound</span>
          <div class="timing-value">&le; {_display_ms(timing["worst_case_bound"]["total_pipeline_ms"])}</div>
        </article>
      </div>
      <p class="timing-meta">
        Representative probe: {escape(_display_probe_shape(timing["probe_shape"]))} ({_display_voxel_count(timing["probe_voxel_count"])} voxels).
        Median of {timing["sample_runs"]} runs after {timing["warmup_runs"]} warm-up runs.
        Backend: {escape(timing["backend"])}. Representative total pipeline: {_display_ms(timing["total_pipeline_ms"])}.
      </p>
      <div class="performance-grid">
        <article class="performance-card">
          <h2>Profiled Bounds</h2>
          <p class="performance-copy">
            High-load profiled bound: &le; {_display_ms(timing["worst_case_bound"]["total_pipeline_ms"])}
            total at {escape(_display_probe_shape(timing["worst_case_bound"]["probe_shape"]))}
            ({_display_voxel_count(timing["worst_case_bound"]["probe_voxel_count"])} voxels),
            with reconstruction bounded at &le; {_display_ms(timing["worst_case_bound"]["reconstruction_ms"])}
            and verification bounded at &le; {_display_ms(timing["worst_case_bound"]["verification_ms"])}.
          </p>
          <p class="performance-note">{escape(timing["worst_case_bound"]["basis"])}</p>
          <ul class="performance-list">
            {benchmark_markup}
          </ul>
        </article>
        <article class="performance-card">
          <h2>Streaming / Partial Validation</h2>
          <p class="performance-copy">{escape(timing["streaming_strategy"]["release_policy"])}</p>
          <ul class="performance-list">
            {streaming_markup}
          </ul>
          <p class="performance-note">
            Early block conditions: {escape(", ".join(timing["streaming_strategy"]["early_block_conditions"]))}.
          </p>
        </article>
      </div>
      <div class="tier-grid">
        {tier_cards_markup}
      </div>
      <div class="failure-banner">
        <h2>{escape(verification["headline"])}</h2>
        <p class="failure-reason">Reason: {escape(verification["primary_reason_summary"])}</p>
        <ul class="failure-list">
          {reason_markup}
        </ul>
      </div>
      <div class="integration-map">
        <article class="integration-node">
          <span class="integration-kicker">Ingress</span>
          <h2>{escape(view_model["integration"]["ingress_label"])}</h2>
          <p>{escape(view_model["integration"]["ingress_detail"])}</p>
        </article>
        <div class="integration-arrow">-&gt;</div>
        <article class="integration-node">
          <span class="integration-kicker">Proxy</span>
          <h2>{escape(view_model["integration"]["proxy_label"])}</h2>
          <p>{escape(view_model["integration"]["proxy_detail"])}</p>
        </article>
        <div class="integration-arrow">-&gt;</div>
        <article class="integration-node">
          <span class="integration-kicker">Egress</span>
          <h2>{escape(view_model["integration"]["egress_label"])}</h2>
          <p>{escape(view_model["integration"]["egress_detail"])}</p>
        </article>
      </div>
      <div class="integration-story-grid">
        <article class="integration-story-card">
          <h2>{escape(view_model["integration"]["dicom_heading"])}</h2>
          <p>{escape(view_model["integration"]["dicom_detail"])}</p>
        </article>
        <article class="integration-story-card">
          <h2>{escape(view_model["integration"]["placement_heading"])}</h2>
          <p>{escape(view_model["integration"]["placement_detail"])}</p>
        </article>
        <article class="integration-story-card">
          <h2>{escape(view_model["integration"]["failure_mode_heading"])}</h2>
          <p><strong>{escape(view_model["integration"]["failure_mode_label"])}</strong>. {escape(view_model["integration"]["failure_mode_detail"])}</p>
          <p class="footnote">{escape(view_model["integration"]["service_outage_detail"])}</p>
        </article>
      </div>
      <div class="metric-grid">
        <article class="metric-card">
          <span class="metric-label">Mask Delta</span>
          <div class="metric-value">{summary["mask_voxel_delta"]}</div>
          <div class="metric-note">Voxel delta between the AI mask and the profile-verified mask.</div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Rejected Regions</span>
          <div class="metric-value">{summary["rejected_region_count"]}</div>
          <div class="metric-note">AI mask voxels that the verification layer removed before PACS/workstation release.</div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Intensity Delta</span>
          <div class="metric-value">{summary["intensity_mismatch_voxel_count"]}</div>
          <div class="metric-note">AI image voxels that diverge from the raw-derived reconstruction.</div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Verified Volume</span>
          <div class="metric-value">{summary["verified_tumor_volume_mm3"]} mm3</div>
          <div class="metric-note">Claimed volume: {summary["claimed_tumor_volume_mm3"]} mm3.</div>
        </article>
      </div>
      <ul class="findings">
        {findings_markup}
      </ul>
    </section>

    <section class="legend">
      <h2>Legend</h2>
      <div class="legend-items">
        <span class="legend-item"><span class="swatch swatch-ai"></span> AI tumour overlay</span>
        <span class="legend-item"><span class="swatch swatch-verified"></span> Verified tumour overlay</span>
        <span class="legend-item"><span class="swatch swatch-mismatch"></span> Intensity mismatch</span>
        <span class="legend-item"><span class="swatch swatch-rejected"></span> Rejected AI region</span>
      </div>
      <p class="footnote">
        Left panels show exactly what the AI attempted to hand downstream. Right panels show what the Guardian Layer can pass through to PACS/workstation:
        reconstruction recomputed from raw k-space and a tumour mask derived by deterministic policy from measured reality.
      </p>
    </section>

    <section class="slice-grid" id="slice-grid"></section>

    <section class="differences">
      <h2>Difference Register</h2>
      {differences_markup}
    </section>
  </main>
  <script>
    const comparisonData = {data_json};

    function createSliceCard(slice) {{
      const card = document.createElement("article");
      card.className = "slice-card";
      card.innerHTML = `
        <header class="slice-header">
          <div>
            <h2 class="slice-title">${{slice.slice_label}}</h2>
          </div>
          <div class="slice-stats">
            <span class="slice-stat">${{slice.mask_delta}} mask delta</span>
            <span class="slice-stat">${{slice.rejected_count}} rejected regions</span>
            <span class="slice-stat">${{slice.intensity_mismatch_count}} intensity mismatches</span>
          </div>
        </header>
        <div class="panel-grid">
          <section class="panel">
            <h3>${{comparisonData.ai_label}}</h3>
            <p>AI-proposed image and mask, shown before the verification layer accepts or rejects the handoff.</p>
            <canvas data-side="ai"></canvas>
          </section>
          <section class="panel">
            <h3>${{comparisonData.verified_label}}</h3>
            <p>Guardian-layer reconstruction from raw k-space with the profile-verified tumour mask.</p>
            <canvas data-side="verified"></canvas>
          </section>
        </div>
      `;
      const aiCanvas = card.querySelector('canvas[data-side="ai"]');
      const verifiedCanvas = card.querySelector('canvas[data-side="verified"]');
      paintSlice(aiCanvas, slice, "ai");
      paintSlice(verifiedCanvas, slice, "verified");
      return card;
    }}

    function computeCellSize(rows, cols) {{
      const longestEdge = Math.max(rows, cols);
      return Math.max(14, Math.min(28, Math.floor(360 / longestEdge)));
    }}

    function normaliseIntensity(value) {{
      const min = comparisonData.intensity_window.min;
      const max = comparisonData.intensity_window.max;
      if (max === min) {{
        return 0.5;
      }}
      return Math.max(0, Math.min(1, (value - min) / (max - min)));
    }}

    function paintSlice(canvas, slice, side) {{
      const rows = slice.ai_reconstruction.length;
      const cols = slice.ai_reconstruction[0].length;
      const cell = computeCellSize(rows, cols);
      canvas.width = cols * cell;
      canvas.height = rows * cell;
      const ctx = canvas.getContext("2d");
      const image = side === "ai" ? slice.ai_reconstruction : slice.verified_reconstruction;
      const mask = side === "ai" ? slice.ai_mask : slice.verified_mask;
      const intensityMismatch = slice.intensity_mismatch_mask;
      const rejected = slice.rejected_mask;

      ctx.fillStyle = "#090d12";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let row = 0; row < rows; row += 1) {{
        for (let col = 0; col < cols; col += 1) {{
          const x = col * cell;
          const y = row * cell;
          const intensity = normaliseIntensity(image[row][col]);
          const shade = Math.round(intensity * 255);
          ctx.fillStyle = `rgb(${{shade}}, ${{shade}}, ${{shade}})`;
          ctx.fillRect(x, y, cell, cell);

          if (mask[row][col]) {{
            ctx.fillStyle = side === "ai" ? "rgba(0, 214, 255, 0.30)" : "rgba(54, 226, 143, 0.32)";
            ctx.fillRect(x, y, cell, cell);
          }}

          if (side === "ai" && rejected[row][col]) {{
            ctx.fillStyle = "rgba(255, 90, 95, 0.18)";
            ctx.fillRect(x, y, cell, cell);
            ctx.strokeStyle = "rgba(255, 90, 95, 0.92)";
            ctx.lineWidth = Math.max(2, Math.floor(cell * 0.12));
            ctx.beginPath();
            ctx.moveTo(x + 2, y + 2);
            ctx.lineTo(x + cell - 2, y + cell - 2);
            ctx.moveTo(x + cell - 2, y + 2);
            ctx.lineTo(x + 2, y + cell - 2);
            ctx.stroke();
          }}

          if (intensityMismatch[row][col]) {{
            ctx.strokeStyle = "rgba(255, 194, 75, 0.98)";
            ctx.lineWidth = Math.max(2, Math.floor(cell * 0.1));
            ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
          }}

          ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, cell, cell);
        }}
      }}
    }}

    const sliceRoot = document.getElementById("slice-grid");
    comparisonData.slices.forEach((slice) => {{
      sliceRoot.appendChild(createSliceCard(slice));
    }});
  </script>
</body>
</html>
"""


def _array_to_nested_lists(values: np.ndarray) -> list[list[float]]:
    return [_array_row_to_list(row) for row in np.asarray(values, dtype=np.float64)]


def _array_row_to_list(values: np.ndarray) -> list[float]:
    return [_round_float(float(value)) for value in np.asarray(values, dtype=np.float64)]


def _difference_rows(
    *,
    ai_reconstruction: np.ndarray,
    verified_reconstruction: np.ndarray,
    ai_mask: np.ndarray,
    verified_mask: np.ndarray,
    intensity_mismatch_mask: np.ndarray,
    rejected_mask: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    total_count = 0

    def _append_row(
        *,
        slice_index: int,
        row_index: int,
        column_index: int,
        code: str,
        status_label: str,
        ai_value: float,
        verified_value: float,
        delta: float,
        ai_masked: bool,
        verified_masked: bool,
    ) -> None:
        nonlocal total_count
        total_count += 1
        if len(rows) >= _MAX_DIFFERENCE_ROWS:
            return
        rows.append(
            {
                "slice_index": int(slice_index),
                "row_index": int(row_index),
                "column_index": int(column_index),
                "code": code,
                "status_label": status_label,
                "ai_value": _round_float(ai_value),
                "verified_value": _round_float(verified_value),
                "delta": _round_float(delta),
                "ai_masked": bool(ai_masked),
                "verified_masked": bool(verified_masked),
            }
        )

    for slice_index, row_index, column_index in np.argwhere(rejected_mask):
        _append_row(
            slice_index=int(slice_index),
            row_index=int(row_index),
            column_index=int(column_index),
            code="mask_false_positive",
            status_label="Rejected Region",
            ai_value=float(ai_reconstruction[slice_index, row_index, column_index]),
            verified_value=float(verified_reconstruction[slice_index, row_index, column_index]),
            delta=float(
                ai_reconstruction[slice_index, row_index, column_index]
                - verified_reconstruction[slice_index, row_index, column_index]
            ),
            ai_masked=True,
            verified_masked=False,
        )
    for slice_index, row_index, column_index in np.argwhere(missing_mask):
        _append_row(
            slice_index=int(slice_index),
            row_index=int(row_index),
            column_index=int(column_index),
            code="mask_false_negative",
            status_label="Missing Verified Region",
            ai_value=float(ai_reconstruction[slice_index, row_index, column_index]),
            verified_value=float(verified_reconstruction[slice_index, row_index, column_index]),
            delta=float(
                ai_reconstruction[slice_index, row_index, column_index]
                - verified_reconstruction[slice_index, row_index, column_index]
            ),
            ai_masked=False,
            verified_masked=True,
        )
    for slice_index, row_index, column_index in np.argwhere(intensity_mismatch_mask):
        _append_row(
            slice_index=int(slice_index),
            row_index=int(row_index),
            column_index=int(column_index),
            code="intensity_mismatch",
            status_label="Intensity Mismatch",
            ai_value=float(ai_reconstruction[slice_index, row_index, column_index]),
            verified_value=float(verified_reconstruction[slice_index, row_index, column_index]),
            delta=float(
                ai_reconstruction[slice_index, row_index, column_index]
                - verified_reconstruction[slice_index, row_index, column_index]
            ),
            ai_masked=bool(ai_mask[slice_index, row_index, column_index]),
            verified_masked=bool(verified_mask[slice_index, row_index, column_index]),
        )
    return rows, total_count


def _build_verification_summary(
    *,
    report: dict[str, Any],
    reasons: list[dict[str, Any]],
    tiers: dict[str, Any],
    release_eligible: bool,
) -> dict[str, Any]:
    ordered_tiers = _ordered_tiers(tiers)
    tier_view_models = [_tier_view_model(tier) for tier in ordered_tiers]
    decision = report["decision"]
    headline = _verification_headline(report)
    primary_reason = _viewer_primary_reason_summary(report)
    if decision["decision_class"] == "blocked_structural":
        primary_reason_entry = reasons[0] if reasons else {
            "summary": "structural_intake_block",
            "message": "Guardian structural intake blocked proof construction.",
        }
        return {
            "headline": headline,
            "primary_reason_summary": primary_reason_entry["summary"],
            "reasons": [
                {"code": reason["code"], "message": reason["message"]}
                for reason in reasons
            ],
            "tiers": tier_view_models,
        }
    if decision["decision_class"] == "blocked_physical":
        primary_reason_entry = reasons[0] if reasons else {
            "summary": "verification_blocked_by_structural_or_physical_gate",
            "message": "Blocking proof verification failed.",
        }
        return {
            "headline": headline,
            "primary_reason_summary": primary_reason_entry["summary"],
            "reasons": [
                {"code": reason["code"], "message": reason["message"]}
                for reason in reasons
            ],
            "tiers": tier_view_models,
        }
    if _report_is_triage_only(report):
        return {
            "headline": headline,
            "primary_reason_summary": primary_reason,
            "reasons": [
                {
                    "code": "proof_passed",
                    "message": (
                        "The deterministic proof path passed for this study under the selected runtime profile."
                    ),
                },
                {
                    "code": "non_final_profile_selected",
                    "message": (
                        "The active runtime profile is triage-only, so no final release or portable bundle is allowed."
                    ),
                }
            ],
            "tiers": tier_view_models,
        }
    if release_eligible:
        return {
            "headline": headline,
            "primary_reason_summary": primary_reason,
            "reasons": _decision_reason_items(decision),
            "tiers": tier_view_models,
        }
    if reasons:
        return {
            "headline": headline,
            "primary_reason_summary": primary_reason,
            "reasons": [
                {"code": reason["code"], "message": reason["message"]}
                for reason in reasons
            ],
            "tiers": tier_view_models,
        }
    return {
        "headline": headline,
        "primary_reason_summary": primary_reason,
        "reasons": _decision_reason_items(decision),
        "tiers": tier_view_models,
    }


def _build_findings(
    *,
    report: dict[str, Any],
    reasons: list[dict[str, Any]],
    tiers: dict[str, Any],
    mask_delta: int,
    rejected_count: int,
    intensity_mismatch_count: int,
    claimed_volume: float,
    computed_volume: float,
    volume_verified: bool,
) -> list[str]:
    clinical_tier = tiers["level_3"]
    decision = report["decision"]
    if _report_is_triage_only(report):
        return [
            "Deterministic proof passed, but the selected runtime profile is explicitly triage-only.",
            "No final PACS/workstation release is allowed under this profile.",
        ]
    if not reasons:
        findings = [
            "Blocking verification passed. Level 1 structural validity and Level 2 physical consistency both held.",
            "No mask drift was detected between the AI overlay and the profile-derived verified mask.",
            "No intensity manipulation was detected in the AI reconstruction.",
        ]
        if clinical_tier["status"] in {"warning", "advisory"}:
            findings.append(
                "Clinical plausibility raised advisory review flags even though the blocking verification gate passed."
            )
        if decision["decision_class"] == "verified_boundary_hugging":
            findings.append(
                "At least one eligible proof metric remained within tolerance but inside the configured guard band of a failure threshold."
            )
        if decision["decision_class"] == "verified_with_advisory":
            findings.append(
                "The released case carries non-blocking advisory findings recorded in the canonical decision layer."
            )
            if decision["success_modifiers"]["near_proof_threshold"]:
                findings.append(
                    "One or more eligible proof metrics also remained within tolerance but inside the configured guard band."
                )
        return findings
    findings: list[str] = []
    if mask_delta:
        findings.append(
            f"Mask verification failed: {mask_delta} voxel(s) differ between the AI mask and the verified mask."
        )
    if rejected_count:
        findings.append(
            f"{rejected_count} AI mask voxel(s) were explicitly rejected because measured reality does not support them."
        )
    if intensity_mismatch_count:
        findings.append(
            f"Intensity verification failed: {intensity_mismatch_count} voxel(s) in the AI image do not match the raw-derived reconstruction."
        )
    if volume_verified:
        findings.append(
            f"The claimed tumour volume ({claimed_volume} mm3) still matches the verified mask volume ({computed_volume} mm3)."
        )
    else:
        findings.append(
            f"Volume verification failed: claimed {claimed_volume} mm3, verified {computed_volume} mm3."
        )
    if clinical_tier["status"] in {"warning", "advisory"}:
        findings.append(
            "Clinical plausibility heuristics also raised advisory review flags on the verified anatomy."
        )
    return findings


def _build_clinical_summary(
    *,
    state: dict[str, Any],
    analysis: dict[str, Any],
    verification: dict[str, Any],
    release_eligible: bool,
    report: dict[str, Any],
) -> dict[str, str]:
    verified_volume_mm3 = analysis["computed_tumor_volume_mm3"]
    decision_class = report["decision"]["decision_class"]
    if _report_is_triage_only(report):
        status_label = "TRIAGE ONLY"
        status_note = "Proof passed, but this runtime profile is non-final and cannot release downstream."
    else:
        status_label = "VERIFIED" if release_eligible else "NOT VERIFIED"
        status_note = (
            "Accepted for downstream PACS/workstation release."
            if release_eligible
            else f"Blocked before downstream handoff. Reason: {verification['primary_reason_summary']}."
        )
    if _report_is_triage_only(report):
        confidence_label = "Derived from raw data (triage only)"
        confidence_note = "The measured-data proof path passed, but the profile is limited to pre-screening."
    elif release_eligible and decision_class == "verified_with_advisory":
        confidence_label = "Derived from raw data (not AI)"
        confidence_note = (
            "Not AI-authored. Released from the measured MRI path with non-blocking advisory findings recorded for review."
        )
    elif release_eligible and decision_class == "verified_boundary_hugging":
        confidence_label = "Derived from raw data (not AI)"
        confidence_note = (
            "Not AI-authored. Released from the measured MRI path, but at least one eligible proof metric sat inside the configured guard band."
        )
    elif release_eligible:
        confidence_label = "Derived from raw data (not AI)"
        confidence_note = "Not AI-authored. Released only from the measured MRI data path."
    else:
        confidence_label = "Not released from AI claim"
        confidence_note = "The raw-data proof path did not support downstream release."
    audit_hash = hash_canonical_json(
        {
            "case_id": state["case_id"],
            "raw_hash": analysis["raw_hash"],
            "reconstruction_hash": analysis["derived_reconstruction_hash"],
            "verified_mask_hash": analysis["verified_mask_hash"],
            "verified_tumor_volume_mm3": verified_volume_mm3,
        }
    ).upper()
    return {
        "status_state": (
            "triage"
            if _report_is_triage_only(report)
            else "verified" if release_eligible else "blocked"
        ),
        "status_label": status_label,
        "status_note": status_note,
        "tumor_volume_cm3": _format_volume_cm3(verified_volume_mm3),
        "tumor_volume_note": f"Verified volume on measured grid: {verified_volume_mm3} mm3.",
        "confidence_label": confidence_label,
        "confidence_note": confidence_note,
        "audit_id": _short_audit_id(audit_hash),
        "audit_note": "Deterministic audit fingerprint for this verified handoff state.",
    }


def _build_integration_summary(
    *,
    release_eligible: bool,
    report: dict[str, Any],
    primary_reason: str,
) -> dict[str, str]:
    decision_class = report["decision"]["decision_class"]
    if _report_is_triage_only(report):
        decision = "blocked"
    else:
        decision = "released" if release_eligible else "blocked"
    narrative = build_guardian_integration_narrative(
        current_guardian_decision=decision,
    )
    if _report_is_triage_only(report):
        egress_label = "No Final PACS Output"
        egress_detail = "This profile is triage-only, so no final downstream series are released."
    else:
        egress_label = "Mock PACS Output" if release_eligible else "PACS Output Blocked"
        if release_eligible and decision_class == "verified_with_advisory":
            egress_detail = (
                "Verified reconstruction, verified segmentation, and evidence records are ready for downstream PACS/workstation handoff, with non-blocking advisory findings preserved in the canonical decision."
            )
        elif release_eligible and decision_class == "verified_boundary_hugging":
            egress_detail = (
                "Verified reconstruction, verified segmentation, and evidence records are ready for downstream PACS/workstation handoff, with guard-band boundary proximity recorded in the canonical decision."
            )
        else:
            egress_detail = (
                "Verified reconstruction, verified segmentation, and evidence records are ready for downstream PACS/workstation handoff."
                if release_eligible
                else f"No downstream series are released. Block reason: {primary_reason}."
            )
    return {
        "ingress_label": "Mock DICOM Input",
        "ingress_detail": (
            "Raw k-space, AI reconstruction, segmentation, and tumour-volume claim arrive over a mocked DICOM boundary."
        ),
        "proxy_label": "MRI Guardian Layer",
        "proxy_detail": (
            "The verification proxy recomputes the image from raw data, applies Level 1/2/3 checks, and decides whether the PACS handoff is allowed."
        ),
        "egress_label": egress_label,
        "egress_detail": egress_detail,
        "dicom_heading": "DICOM Compliance",
        "dicom_detail": narrative["dicom_compliance_story"]["story"],
        "placement_heading": "Hospital Placement",
        "placement_detail": narrative["hospital_placement"]["story"],
        "failure_mode_heading": "Failure Behaviour",
        "failure_mode_label": "Fail-closed verification handoff",
        "failure_mode_detail": narrative["failure_mode"]["on_verification_failure"],
        "service_outage_detail": narrative["failure_mode"]["on_service_unavailable"],
    }


def _format_volume_cm3(volume_mm3: float) -> str:
    volume_cm3 = volume_mm3 / 1000.0
    if volume_cm3 >= 1:
        return f"{volume_cm3:.3f}".rstrip("0").rstrip(".")
    if volume_cm3 >= 0.1:
        return f"{volume_cm3:.4f}".rstrip("0").rstrip(".")
    return f"{volume_cm3:.5f}".rstrip("0").rstrip(".")


def _short_audit_id(audit_hash: str) -> str:
    return f"{audit_hash[:4]}...{audit_hash[-3:]}"


def _report_is_triage_only(report: dict[str, Any]) -> bool:
    return any(reason["code"] == "non_final_profile_selected" for reason in report["failure_reasons"])


def _decision_reason_items(decision: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"code": item["code"], "message": item["message"]}
        for item in decision["decision_reasons"]
    ]


def _viewer_primary_reason_summary(report: dict[str, Any]) -> str:
    reasons = report["failure_reasons"]
    if reasons:
        return str(reasons[0]["summary"])
    decision = report["decision"]
    if decision["reason_codes"]:
        return str(decision["reason_codes"][0])
    return "final_release_authorized" if report["release_eligible"] else "decision_blocked"


def _verification_headline(report: dict[str, Any]) -> str:
    decision_class = report["decision"]["decision_class"]
    if decision_class == "blocked_structural":
        return "Structural BLOCKED"
    if decision_class == "blocked_physical":
        return "Verification FAILED"
    if _report_is_triage_only(report):
        return "TRIAGE ONLY"
    if decision_class == "blocked_policy":
        return "Profile Policy BLOCKED"
    if decision_class == "verified_with_advisory":
        return "Verification PASSED WITH ADVISORY"
    if decision_class == "verified_boundary_hugging":
        return "Verification PASSED NEAR BOUNDARY"
    return "Verification PASSED"


def _ordered_tiers(tiers: dict[str, Any]) -> list[dict[str, Any]]:
    return [tiers[key] for key in ("level_1", "level_2", "level_3")]


def _tier_view_model(tier: dict[str, Any]) -> dict[str, Any]:
    status = tier["status"]
    status_label_map = {
        "passed": "PASS" if tier["blocking"] else "ADVISORY PASS",
        "failed": "FAIL",
        "advisory": "ADVISORY NOTE",
        "warning": "ADVISORY WARN",
        "not_evaluated": "NOT EVALUATED",
    }
    status_class_map = {
        "passed": "tier-status-pass",
        "failed": "tier-status-fail",
        "advisory": "tier-status-warn",
        "warning": "tier-status-warn",
        "not_evaluated": "tier-status-warn",
    }
    return {
        **tier,
        "scope_label": "Blocking gate" if tier["blocking"] else "Advisory only",
        "status_label": status_label_map.get(status, status.upper()),
        "status_class": status_class_map.get(status, "tier-status-warn"),
    }


def _tier_cards_markup(tiers: list[dict[str, Any]]) -> str:
    cards = []
    for tier in tiers:
        checks_markup = "\n".join(
            (
                "<li>"
                f"<strong>{escape(check['label'])}</strong>: {escape(check['message'])}"
                "</li>"
            )
            for check in tier["checks"]
        )
        cards.append(
            "<article class=\"tier-card\">"
            "<div class=\"tier-header\">"
            "<div>"
            f"<h2 class=\"tier-title\">{escape(tier['title'])}</h2>"
            f"<p class=\"tier-subtitle\">{escape(tier['subtitle'])}</p>"
            "</div>"
            f"<span class=\"tier-status {escape(tier['status_class'])}\">{escape(tier['status_label'])}</span>"
            "</div>"
            f"<p class=\"tier-scope\">{escape(tier['scope_label'])}</p>"
            f"<p class=\"tier-summary\">{escape(tier['summary'])}</p>"
            f"<ul class=\"tier-checks\">{checks_markup}</ul>"
            "</article>"
        )
    return "\n".join(cards)


def _difference_table_markup(differences: dict[str, Any]) -> str:
    rows = differences["rows"]
    if not rows:
        return (
            "<p class=\"footnote\">No difference rows were recorded. The AI image and tumour overlay "
            "match the deterministic verification path.</p>"
        )
    body_rows = []
    for row in rows:
        chip_class = {
            "Rejected Region": "chip-rejected",
            "Missing Verified Region": "chip-missing",
            "Intensity Mismatch": "chip-mismatch",
        }[row["status_label"]]
        body_rows.append(
            "<tr>"
            f"<td>{row['slice_index'] + 1}</td>"
            f"<td>{row['row_index']}</td>"
            f"<td>{row['column_index']}</td>"
            f"<td>{escape(row['code'])}</td>"
            f"<td>{row['ai_value']}</td>"
            f"<td>{row['verified_value']}</td>"
            f"<td>{row['delta']}</td>"
            f"<td>{'Yes' if row['ai_masked'] else 'No'}</td>"
            f"<td>{'Yes' if row['verified_masked'] else 'No'}</td>"
            f"<td><span class=\"chip {chip_class}\">{escape(row['status_label'])}</span></td>"
            "</tr>"
        )
    omitted_count = differences["omitted_count"]
    omitted_note = ""
    if omitted_count:
        omitted_note = (
            f"<p class=\"footnote\">{omitted_count} additional difference row(s) are omitted "
            "from this table for readability.</p>"
        )
    return (
        "<table>"
        "<thead><tr><th>Slice</th><th>Row</th><th>Column</th><th>Code</th><th>AI</th>"
        "<th>Verified</th><th>Delta</th><th>AI Masked</th><th>Verified Masked</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        f"{omitted_note}"
    )


def _benchmark_list_markup(timing: dict[str, Any]) -> str:
    items = []
    for benchmark in timing["benchmarks"]:
        items.append(
            "<li>"
            f"<strong>{escape(benchmark['label'])}</strong>: "
            f"{escape(_display_probe_shape(benchmark['probe_shape']))} "
            f"({_display_voxel_count(benchmark['probe_voxel_count'])} voxels), "
            f"reconstruction {_display_ms(benchmark['reconstruction_ms'])}, "
            f"verification {_display_ms(benchmark['verification_ms'])}, "
            f"total {_display_ms(benchmark['total_pipeline_ms'])}, "
            f"overhead +{_display_percent(benchmark['overhead_percent'])}. "
            f"{escape(benchmark['purpose'])}"
            "</li>"
        )
    return "\n".join(items)


def _streaming_strategy_markup(streaming_strategy: dict[str, Any]) -> str:
    items = []
    for stage in streaming_strategy["stages"]:
        items.append(
            "<li>"
            f"<strong>{escape(stage['title'])}</strong> ({escape(stage['window'])}): "
            f"{escape(stage['action'])} Benefit: {escape(stage['benefit'])}"
            "</li>"
        )
    return "\n".join(items)


def _display_ms(value: float) -> str:
    rounded = int(round(float(value)))
    return f"{rounded}ms"


def _display_percent(value: float) -> str:
    rounded = int(round(float(value)))
    return f"{rounded}%"


def _display_probe_shape(shape: list[int]) -> str:
    return "x".join(str(int(item)) for item in shape)


def _display_voxel_count(value: int) -> str:
    return f"{int(value):,}"
