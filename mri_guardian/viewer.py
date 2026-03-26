"""Side-by-side visual report for AI output versus verified MRI reality."""

from __future__ import annotations

import json
from html import escape
from typing import Any

import numpy as np

from .harness import (
    MRI_GUARDIAN_POLICY_NAME,
    MRI_GUARDIAN_POLICY_VERSION,
    _RECONSTRUCTION_TOLERANCE,
    _VOLUME_TOLERANCE,
    _analyse_case,
    _build_verified_manifest,
    _extract_arrays,
    _round_float,
)

_MAX_DIFFERENCE_ROWS = 64


def build_side_by_side_view_model(state: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic model for side-by-side AI versus verified review."""
    arrays, failure = _extract_arrays(state)
    if failure is not None:
        raise ValueError(failure.message)
    analysis = _analyse_case(state)
    manifest = _build_verified_manifest(state, analysis)
    ai_reconstruction = np.asarray(arrays["ai_reconstruction"], dtype=np.float64)
    ai_mask = np.asarray(arrays["tumor_mask"], dtype=np.int64)
    verified_reconstruction = np.asarray(analysis["derived_reconstruction"], dtype=np.float64)
    mismatch_mask = np.abs(ai_reconstruction - verified_reconstruction) > _RECONSTRUCTION_TOLERANCE
    rejected_mask = ((ai_mask == 1) & mismatch_mask).astype(np.int64)
    verified_mask = np.where(rejected_mask == 1, 0, ai_mask).astype(np.int64)
    claimed_volume = _round_float(float(state["tumor_volume_mm3"]))
    computed_volume = analysis["computed_tumor_volume_mm3"]
    volume_verified = abs(claimed_volume - computed_volume) <= _VOLUME_TOLERANCE

    mismatch_count = int(np.count_nonzero(mismatch_mask))
    rejected_count = int(np.count_nonzero(rejected_mask))
    verified_voxel_count = int(np.count_nonzero(verified_mask))
    intensity_min = _round_float(float(min(ai_reconstruction.min(), verified_reconstruction.min())))
    intensity_max = _round_float(float(max(ai_reconstruction.max(), verified_reconstruction.max())))

    slices: list[dict[str, Any]] = []
    for slice_index in range(ai_reconstruction.shape[0]):
        slice_mismatch_mask = mismatch_mask[slice_index]
        slice_rejected_mask = rejected_mask[slice_index]
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
                "mismatch_mask": slice_mismatch_mask.astype(np.int64).tolist(),
                "rejected_mask": slice_rejected_mask.astype(np.int64).tolist(),
                "mismatch_count": int(np.count_nonzero(slice_mismatch_mask)),
                "rejected_count": int(np.count_nonzero(slice_rejected_mask)),
            }
        )

    difference_rows = _difference_rows(
        ai_reconstruction=ai_reconstruction,
        verified_reconstruction=verified_reconstruction,
        ai_mask=ai_mask,
        mismatch_mask=mismatch_mask,
        rejected_mask=rejected_mask,
    )
    findings = _build_findings(
        mismatch_count=mismatch_count,
        rejected_count=rejected_count,
        volume_verified=volume_verified,
        claimed_volume=claimed_volume,
        computed_volume=computed_volume,
    )
    return {
        "title": "The Verifiable MRI Guardian",
        "subtitle": "AI claim versus measured reality on the radiologist boundary.",
        "policy_name": MRI_GUARDIAN_POLICY_NAME,
        "policy_version": MRI_GUARDIAN_POLICY_VERSION,
        "ai_label": "AI Reconstruction + Tumour Mask",
        "verified_label": "Deterministic Reconstruction + Verified Mask",
        "release_eligible": all(
            manifest[section]["verified"]
            for section in (
                "measurement_consistency",
                "ai_reconstruction_match",
                "tumor_volume_proof",
                "no_invented_pixels",
            )
        ),
        "findings": findings,
        "summary": {
            "mismatch_voxel_count": mismatch_count,
            "rejected_region_count": rejected_count,
            "verified_mask_voxel_count": verified_voxel_count,
            "claimed_tumor_volume_mm3": claimed_volume,
            "verified_tumor_volume_mm3": computed_volume,
            "volume_verified": volume_verified,
            "measurement_consistency_verified": manifest["measurement_consistency"]["verified"],
            "max_ai_abs_error": manifest["ai_reconstruction_match"]["max_abs_error"],
            "max_measurement_abs_error": manifest["measurement_consistency"]["max_abs_error"],
        },
        "intensity_window": {
            "min": intensity_min,
            "max": intensity_max,
        },
        "differences": {
            "rows": difference_rows,
            "omitted_count": max(0, mismatch_count - len(difference_rows)),
        },
        "slices": slices,
    }


def render_side_by_side_report_html(
    state: dict[str, Any],
    *,
    report_title: str | None = None,
) -> str:
    """Render a self-contained HTML report for human review."""
    view_model = build_side_by_side_view_model(state)
    title = report_title if report_title is not None else view_model["title"]
    summary = view_model["summary"]
    findings_markup = "\n".join(
        f"<li>{escape(finding)}</li>" for finding in view_model["findings"]
    )
    differences_markup = _difference_table_markup(view_model["differences"])
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
      margin-bottom: 24px;
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

    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
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
      <p class="eyebrow">ICSD Visual Evidence</p>
      <h1>{escape(title)}</h1>
      <p class="subtitle">{escape(view_model["subtitle"])}</p>
      <div class="status-row">
        <span class="status-pill {'status-pass' if view_model['release_eligible'] else 'status-fail'}">
          {'Release Eligible' if view_model['release_eligible'] else 'Release Rejected'}
        </span>
        <span class="status-pill">Policy {escape(view_model["policy_name"])} v{escape(view_model["policy_version"])}</span>
      </div>
      <div class="metric-grid">
        <article class="metric-card">
          <span class="metric-label">Mismatched Voxels</span>
          <div class="metric-value">{summary["mismatch_voxel_count"]}</div>
          <div class="metric-note">AI pixels that do not agree with raw-derived reconstruction.</div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Rejected Regions</span>
          <div class="metric-value">{summary["rejected_region_count"]}</div>
          <div class="metric-note">Masked voxels resting on mismatched image evidence.</div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Volume Claim</span>
          <div class="metric-value">{summary["claimed_tumor_volume_mm3"]} mm³</div>
          <div class="metric-note">Verified from mask geometry: {summary["verified_tumor_volume_mm3"]} mm³.</div>
        </article>
        <article class="metric-card">
          <span class="metric-label">Max AI Error</span>
          <div class="metric-value">{summary["max_ai_abs_error"]}</div>
          <div class="metric-note">Absolute voxel delta against deterministic reconstruction.</div>
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
        <span class="legend-item"><span class="swatch swatch-mismatch"></span> Mismatched voxel</span>
        <span class="legend-item"><span class="swatch swatch-rejected"></span> Rejected region</span>
      </div>
      <p class="footnote">
        Left panels show exactly what the AI tried to send downstream. Right panels show what survives the ICSD guard:
        reconstruction recomputed from raw k-space and tumour overlay trimmed away from any unsupported voxel.
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
            <span class="slice-stat">${{slice.mismatch_count}} mismatched voxels</span>
            <span class="slice-stat">${{slice.rejected_count}} rejected regions</span>
          </div>
        </header>
        <div class="panel-grid">
          <section class="panel">
            <h3>${{comparisonData.ai_label}}</h3>
            <p>AI-proposed image and mask, shown before the guard accepts or rejects them.</p>
            <canvas data-side="ai"></canvas>
          </section>
          <section class="panel">
            <h3>${{comparisonData.verified_label}}</h3>
            <p>Raw-derived reconstruction with only admissible overlay voxels preserved.</p>
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
      return Math.max(22, Math.min(44, Math.floor(360 / longestEdge)));
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
      const mismatch = slice.mismatch_mask;
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

          if (rejected[row][col]) {{
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

          if (mismatch[row][col]) {{
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
    mismatch_mask: np.ndarray,
    rejected_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mismatch_indices = np.argwhere(mismatch_mask)
    for slice_index, row_index, column_index in mismatch_indices[:_MAX_DIFFERENCE_ROWS]:
        ai_value = _round_float(float(ai_reconstruction[slice_index, row_index, column_index]))
        verified_value = _round_float(
            float(verified_reconstruction[slice_index, row_index, column_index])
        )
        rows.append(
            {
                "slice_index": int(slice_index),
                "row_index": int(row_index),
                "column_index": int(column_index),
                "ai_value": ai_value,
                "verified_value": verified_value,
                "delta": _round_float(ai_value - verified_value),
                "ai_masked": bool(ai_mask[slice_index, row_index, column_index]),
                "rejected_region": bool(rejected_mask[slice_index, row_index, column_index]),
            }
        )
    return rows


def _build_findings(
    *,
    mismatch_count: int,
    rejected_count: int,
    volume_verified: bool,
    claimed_volume: float,
    computed_volume: float,
) -> list[str]:
    findings: list[str] = []
    if mismatch_count == 0:
        findings.append("No voxel mismatches detected between the AI image and the raw-derived reconstruction.")
    else:
        findings.append(
            f"{mismatch_count} voxel(s) differ from the deterministic reconstruction and are visually marked."
        )
    if rejected_count == 0:
        findings.append("No tumour regions were rejected by the pixel-origin guard.")
    else:
        findings.append(
            f"{rejected_count} masked voxel(s) were rejected because the supporting pixels do not come from measured reality."
        )
    if volume_verified:
        findings.append(
            f"The claimed tumour volume ({claimed_volume} mm³) matches the verified mask geometry."
        )
    else:
        findings.append(
            f"The claimed tumour volume ({claimed_volume} mm³) does not match the verified mask geometry ({computed_volume} mm³)."
        )
    return findings


def _difference_table_markup(differences: dict[str, Any]) -> str:
    rows = differences["rows"]
    if not rows:
        return (
            "<p class=\"footnote\">No mismatched voxels were recorded. The AI image and the "
            "raw-derived reconstruction are identical at the displayed grid.</p>"
        )
    body_rows = []
    for row in rows:
        chip_class = "chip-rejected" if row["rejected_region"] else "chip-mismatch"
        chip_label = "Rejected Region" if row["rejected_region"] else "Mismatch"
        body_rows.append(
            "<tr>"
            f"<td>{row['slice_index'] + 1}</td>"
            f"<td>{row['row_index']}</td>"
            f"<td>{row['column_index']}</td>"
            f"<td>{row['ai_value']}</td>"
            f"<td>{row['verified_value']}</td>"
            f"<td>{row['delta']}</td>"
            f"<td>{'Yes' if row['ai_masked'] else 'No'}</td>"
            f"<td><span class=\"chip {chip_class}\">{chip_label}</span></td>"
            "</tr>"
        )
    omitted_count = differences["omitted_count"]
    omitted_note = ""
    if omitted_count:
        omitted_note = (
            f"<p class=\"footnote\">{omitted_count} additional mismatched voxel(s) are omitted "
            "from this table for readability.</p>"
        )
    return (
        "<table>"
        "<thead><tr><th>Slice</th><th>Row</th><th>Column</th><th>AI</th><th>Verified</th>"
        "<th>Delta</th><th>AI Masked</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        f"{omitted_note}"
    )
