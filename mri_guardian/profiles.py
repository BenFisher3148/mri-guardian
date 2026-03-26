"""Guardian runtime profile registry and descriptor hashing helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from icsd.core.invariants import Invariant, InvariantProfile, InvariantProfileRegistry, InvariantSet
from icsd.core.serialise import canonical_json_dumps, hash_canonical_json

PROFILE_SCHEMA_VERSION = "guardian-profile/v1"
PROFILE_REVISION_SCOPE = "profile_name"
PROFILE_DIGEST_ALGORITHM = "sha256"
PROFILE_CANONICALIZATION = "icsd.canonical_json/v1"
PROFILE_DIGEST_SCOPE = "verification_profile_descriptor_without_digest_fields"
DEFAULT_PROFILE_NAME = "brain_t1_baseline"
DEFAULT_PROFILE_REVISION = "1"
DEFAULT_ICSD_POLICY_VERSION = "2026.03"

HEURISTIC_DISABLED = "disabled"
HEURISTIC_ADVISORY = "advisory"
HEURISTIC_WARNING = "warning"
HEURISTIC_BLOCKING = "blocking"

DECISION_CLASS_FINAL_RELEASE = "final_release_candidate"
DECISION_CLASS_TRIAGE_ONLY = "triage_only"


@dataclass(frozen=True)
class VerificationProfileDescriptor:
    """Canonical Guardian runtime descriptor for one named verification selection."""

    profile_name: str
    profile_revision: str
    sequence_family: str
    clinical_mode: str
    purpose: str
    risk_posture: str
    operator_interpretation: str
    icsd_policy_id: str
    icsd_policy_version: str
    release_semantics: dict[str, Any]
    mask_rules: dict[str, Any]
    tolerance_rules: dict[str, Any]
    authority_requirements: dict[str, Any]
    heuristic_rules: dict[str, Any]
    performance_expectations: dict[str, Any]

    @property
    def authority(self) -> str:
        return f"{self.icsd_policy_id}@{self.icsd_policy_version}"

    @property
    def mask_threshold(self) -> float:
        return float(self.mask_rules["threshold"])

    @property
    def guard_band_fraction(self) -> float:
        return float(self.tolerance_rules["guard_band_fraction"])

    @property
    def required_authority_source_types(self) -> tuple[str, ...]:
        return tuple(self.authority_requirements["required_source_types"])

    def tolerance(self, field_name: str) -> float:
        return float(self.tolerance_rules[field_name])

    def heuristic(self, code: str) -> dict[str, Any]:
        return deepcopy(self.heuristic_rules[code])

    def digestable_dict(self) -> dict[str, Any]:
        return {
            "profile_schema_version": PROFILE_SCHEMA_VERSION,
            "profile_name": self.profile_name,
            "profile_revision": self.profile_revision,
            "profile_revision_scope": PROFILE_REVISION_SCOPE,
            "sequence_family": self.sequence_family,
            "clinical_mode": self.clinical_mode,
            "purpose": self.purpose,
            "risk_posture": self.risk_posture,
            "operator_interpretation": self.operator_interpretation,
            "icsd_policy_id": self.icsd_policy_id,
            "icsd_policy_version": self.icsd_policy_version,
            "release_semantics": deepcopy(self.release_semantics),
            "mask_rules": deepcopy(self.mask_rules),
            "tolerance_rules": deepcopy(self.tolerance_rules),
            "authority_requirements": deepcopy(self.authority_requirements),
            "heuristic_rules": deepcopy(self.heuristic_rules),
            "performance_expectations": deepcopy(self.performance_expectations),
        }

    def profile_digest(self) -> str:
        return hash_canonical_json(self.digestable_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self.digestable_dict()
        payload.update(
            {
                "profile_digest": self.profile_digest(),
                "digest_algorithm": PROFILE_DIGEST_ALGORITHM,
                "canonicalization": PROFILE_CANONICALIZATION,
                "digest_scope": PROFILE_DIGEST_SCOPE,
            }
        )
        return payload


def build_verification_profile_descriptor(profile_name: str) -> dict[str, Any]:
    """Return the canonical serialized descriptor for one runtime profile."""
    return get_verification_profile(profile_name).to_dict()


def canonicalise_verification_profile_descriptor(
    descriptor: VerificationProfileDescriptor | dict[str, Any],
) -> str:
    """Return canonical JSON for a profile descriptor payload."""
    if isinstance(descriptor, VerificationProfileDescriptor):
        payload = descriptor.to_dict()
    else:
        payload = verification_profile_descriptor_from_dict(dict(descriptor)).to_dict()
    return canonical_json_dumps(payload)


def verification_profile_descriptor_from_dict(
    payload: dict[str, Any],
) -> VerificationProfileDescriptor:
    """Hydrate and validate a descriptor from serialized form."""
    required_fields = (
        "profile_schema_version",
        "profile_name",
        "profile_revision",
        "profile_revision_scope",
        "profile_digest",
        "digest_algorithm",
        "canonicalization",
        "digest_scope",
        "sequence_family",
        "clinical_mode",
        "purpose",
        "risk_posture",
        "operator_interpretation",
        "icsd_policy_id",
        "icsd_policy_version",
        "release_semantics",
        "mask_rules",
        "tolerance_rules",
        "authority_requirements",
        "heuristic_rules",
        "performance_expectations",
    )
    missing_fields = [field_name for field_name in required_fields if field_name not in payload]
    if missing_fields:
        raise ValueError(
            "Verification profile descriptor is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )
    if payload["profile_schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported profile_schema_version: "
            f"{payload['profile_schema_version']!r}."
        )
    if payload["profile_revision_scope"] != PROFILE_REVISION_SCOPE:
        raise ValueError(
            "Unsupported profile_revision_scope: "
            f"{payload['profile_revision_scope']!r}."
        )
    if payload["digest_algorithm"] != PROFILE_DIGEST_ALGORITHM:
        raise ValueError(
            "Unsupported digest_algorithm: "
            f"{payload['digest_algorithm']!r}."
        )
    if payload["canonicalization"] != PROFILE_CANONICALIZATION:
        raise ValueError(
            "Unsupported canonicalization: "
            f"{payload['canonicalization']!r}."
        )
    if payload["digest_scope"] != PROFILE_DIGEST_SCOPE:
        raise ValueError(
            "Unsupported digest_scope: "
            f"{payload['digest_scope']!r}."
        )
    descriptor = VerificationProfileDescriptor(
        profile_name=str(payload["profile_name"]),
        profile_revision=str(payload["profile_revision"]),
        sequence_family=str(payload["sequence_family"]),
        clinical_mode=str(payload["clinical_mode"]),
        purpose=str(payload["purpose"]),
        risk_posture=str(payload["risk_posture"]),
        operator_interpretation=str(payload["operator_interpretation"]),
        icsd_policy_id=str(payload["icsd_policy_id"]),
        icsd_policy_version=str(payload["icsd_policy_version"]),
        release_semantics=deepcopy(dict(payload["release_semantics"])),
        mask_rules=deepcopy(dict(payload["mask_rules"])),
        tolerance_rules=deepcopy(dict(payload["tolerance_rules"])),
        authority_requirements=deepcopy(dict(payload["authority_requirements"])),
        heuristic_rules=deepcopy(dict(payload["heuristic_rules"])),
        performance_expectations=deepcopy(dict(payload["performance_expectations"])),
    )
    expected_digest = descriptor.profile_digest()
    observed_digest = str(payload["profile_digest"])
    if expected_digest != observed_digest:
        raise ValueError(
            "Verification profile descriptor digest mismatch: "
            f"expected {expected_digest!r}, got {observed_digest!r}."
        )
    return descriptor


def list_verification_profiles() -> tuple[str, ...]:
    """Return the available Guardian runtime profile names."""
    return tuple(_PROFILES)


def get_verification_profile(profile_name: str) -> VerificationProfileDescriptor:
    """Return a deep-copied runtime profile descriptor by name."""
    try:
        descriptor = _PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"Unknown Guardian runtime profile: {profile_name!r}.") from exc
    return deepcopy(descriptor)


def build_profile_registry(
    invariants: InvariantSet | tuple[Invariant, ...] | list[Invariant],
) -> InvariantProfileRegistry:
    """Bind all runtime profiles onto ICSD invariant profile identities."""
    invariant_set = invariants if isinstance(invariants, InvariantSet) else InvariantSet(invariants)
    return InvariantProfileRegistry(
        tuple(
            InvariantProfile(
                name=descriptor.profile_name,
                policy=descriptor.icsd_policy_id,
                policy_version=descriptor.icsd_policy_version,
                invariants=invariant_set,
            )
            for descriptor in _PROFILES.values()
        )
    )


def build_authority_requirement_map() -> dict[str, tuple[str, ...]]:
    """Return required authority-source types keyed by resolved ICSD authority."""
    return {
        descriptor.authority: descriptor.required_authority_source_types
        for descriptor in _PROFILES.values()
    }


def resolve_profile_name(
    *,
    profile_name: str | None,
    state: dict[str, Any] | None = None,
) -> str:
    """Resolve the active runtime profile name from explicit or embedded state."""
    if isinstance(profile_name, str) and profile_name.strip():
        return profile_name.strip()
    if isinstance(state, dict):
        manifest = state.get("verified_manifest")
        if isinstance(manifest, dict):
            embedded_name = manifest.get("profile_name")
            if isinstance(embedded_name, str) and embedded_name.strip():
                return embedded_name.strip()
        profile_block = state.get("verification_profile_descriptor")
        if isinstance(profile_block, dict):
            embedded_name = profile_block.get("profile_name")
            if isinstance(embedded_name, str) and embedded_name.strip():
                return embedded_name.strip()
    return DEFAULT_PROFILE_NAME


def default_icsd_policy_version() -> str:
    return get_verification_profile(DEFAULT_PROFILE_NAME).icsd_policy_version


def _profile(
    *,
    profile_name: str,
    sequence_family: str,
    clinical_mode: str,
    purpose: str,
    risk_posture: str,
    operator_interpretation: str,
    icsd_policy_id: str,
    threshold: float,
    guard_band_fraction: float,
    spatial_per_voxel_abs_error: float,
    spatial_global_relative_l2_error: float,
    frequency_per_coefficient_abs_error: float,
    frequency_global_relative_l2_error: float,
    imaginary_leakage_abs_error: float,
    volume_absolute_error: float,
    required_authority_source_types: tuple[str, ...],
    heuristic_rules: dict[str, Any],
    representative_total_ms_budget: float,
    worst_case_total_ms_budget: float,
    final_release_allowed: bool,
    portable_bundle_allowed: bool,
    transparency_artifact_allowed: bool,
    requires_operator_review_on_warning: bool,
    success_decision_class: str,
) -> VerificationProfileDescriptor:
    return VerificationProfileDescriptor(
        profile_name=profile_name,
        profile_revision=DEFAULT_PROFILE_REVISION,
        sequence_family=sequence_family,
        clinical_mode=clinical_mode,
        purpose=purpose,
        risk_posture=risk_posture,
        operator_interpretation=operator_interpretation,
        icsd_policy_id=icsd_policy_id,
        icsd_policy_version=DEFAULT_ICSD_POLICY_VERSION,
        release_semantics={
            "final_release_allowed": final_release_allowed,
            "portable_bundle_allowed": portable_bundle_allowed,
            "transparency_artifact_allowed": transparency_artifact_allowed,
            "requires_operator_review_on_warning": requires_operator_review_on_warning,
            "multiple_warnings_allowed": True,
            "warnings_allow_final_release": final_release_allowed,
            "success_decision_class": success_decision_class,
        },
        mask_rules={
            "algorithm": "threshold_on_raw_derived_reconstruction",
            "deterministic": True,
            "input_semantics": "raw_derived_reconstruction",
            "output_semantics": "verified_mask",
            "comparison": "greater_than_or_equal",
            "threshold": float(threshold),
        },
        tolerance_rules={
            "guard_band_fraction": float(guard_band_fraction),
            "spatial_per_voxel_abs_error": float(spatial_per_voxel_abs_error),
            "spatial_global_relative_l2_error": float(spatial_global_relative_l2_error),
            "frequency_per_coefficient_abs_error": float(frequency_per_coefficient_abs_error),
            "frequency_global_relative_l2_error": float(frequency_global_relative_l2_error),
            "imaginary_leakage_abs_error": float(imaginary_leakage_abs_error),
            "volume_absolute_error": float(volume_absolute_error),
        },
        authority_requirements={
            "required_source_types": list(required_authority_source_types),
        },
        heuristic_rules=deepcopy(heuristic_rules),
        performance_expectations={
            "normative": False,
            "representative_total_ms_budget": float(representative_total_ms_budget),
            "worst_case_total_ms_budget": float(worst_case_total_ms_budget),
        },
    )


def _heuristic(
    *,
    label: str,
    severity: str,
    comparison: str,
    threshold: float,
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "label": label,
        "enabled": enabled,
        "severity": severity,
        "comparison": comparison,
        "threshold": float(threshold),
    }


_COMMON_TOLERANCES = {
    "spatial_per_voxel_abs_error": 1e-9,
    "spatial_global_relative_l2_error": 1e-12,
    "frequency_per_coefficient_abs_error": 1e-9,
    "frequency_global_relative_l2_error": 1e-12,
    "imaginary_leakage_abs_error": 1e-9,
    "volume_absolute_error": 1e-9,
}

_COMMON_FINAL_AUTHORITIES = (
    "scanner_manifest",
    "voxel_geometry",
    "model_manifest",
    "sequence_manifest",
)

_PROFILES = {
    "brain_t1_baseline": _profile(
        profile_name="brain_t1_baseline",
        sequence_family="brain_t1",
        clinical_mode="baseline_final_release",
        purpose="Default operational Guardian release path.",
        risk_posture="baseline",
        operator_interpretation="Standard final-release verification posture for non-contrast T1 studies.",
        icsd_policy_id="policy/mri-guardian/brain-t1-baseline",
        threshold=2.5,
        guard_band_fraction=0.8,
        required_authority_source_types=_COMMON_FINAL_AUTHORITIES,
        heuristic_rules={
            "mask_fraction_guardrail": _heuristic(
                label="Verified occupancy fraction",
                severity=HEURISTIC_WARNING,
                comparison="less_than_or_equal",
                threshold=0.35,
            ),
            "inside_outside_contrast": _heuristic(
                label="Inside/outside contrast delta",
                severity=HEURISTIC_WARNING,
                comparison="greater_than_or_equal",
                threshold=0.75,
            ),
            "slice_span": _heuristic(
                label="Slice span",
                severity=HEURISTIC_WARNING,
                comparison="greater_than_or_equal",
                threshold=2.0,
            ),
        },
        representative_total_ms_budget=80.0,
        worst_case_total_ms_budget=180.0,
        final_release_allowed=True,
        portable_bundle_allowed=True,
        transparency_artifact_allowed=True,
        requires_operator_review_on_warning=True,
        success_decision_class=DECISION_CLASS_FINAL_RELEASE,
        **_COMMON_TOLERANCES,
    ),
    "brain_t2_baseline": _profile(
        profile_name="brain_t2_baseline",
        sequence_family="brain_t2",
        clinical_mode="baseline_final_release",
        purpose="Default T2 verification posture.",
        risk_posture="baseline",
        operator_interpretation="Standard final-release verification posture for T2-weighted studies.",
        icsd_policy_id="policy/mri-guardian/brain-t2-baseline",
        threshold=1.75,
        guard_band_fraction=0.8,
        required_authority_source_types=_COMMON_FINAL_AUTHORITIES,
        heuristic_rules={
            "mask_fraction_guardrail": _heuristic(
                label="Verified occupancy fraction",
                severity=HEURISTIC_WARNING,
                comparison="less_than_or_equal",
                threshold=0.35,
            ),
            "inside_outside_contrast": _heuristic(
                label="Inside/outside contrast delta",
                severity=HEURISTIC_WARNING,
                comparison="greater_than_or_equal",
                threshold=0.35,
            ),
            "slice_span": _heuristic(
                label="Slice span",
                severity=HEURISTIC_WARNING,
                comparison="greater_than_or_equal",
                threshold=2.0,
            ),
        },
        representative_total_ms_budget=80.0,
        worst_case_total_ms_budget=180.0,
        final_release_allowed=True,
        portable_bundle_allowed=True,
        transparency_artifact_allowed=True,
        requires_operator_review_on_warning=True,
        success_decision_class=DECISION_CLASS_FINAL_RELEASE,
        **_COMMON_TOLERANCES,
    ),
    "contrast_enhanced_review": _profile(
        profile_name="contrast_enhanced_review",
        sequence_family="brain_t1_post_contrast",
        clinical_mode="contrast_review_final_release",
        purpose="Post-contrast verification with contrast-protocol anchoring.",
        risk_posture="review",
        operator_interpretation="Final-release posture for post-contrast review with explicit protocol anchoring.",
        icsd_policy_id="policy/mri-guardian/contrast-enhanced-review",
        threshold=3.0,
        guard_band_fraction=0.8,
        required_authority_source_types=_COMMON_FINAL_AUTHORITIES + ("contrast_protocol",),
        heuristic_rules={
            "mask_fraction_guardrail": _heuristic(
                label="Verified occupancy fraction",
                severity=HEURISTIC_WARNING,
                comparison="less_than_or_equal",
                threshold=0.30,
            ),
            "inside_outside_contrast": _heuristic(
                label="Inside/outside contrast delta",
                severity=HEURISTIC_WARNING,
                comparison="greater_than_or_equal",
                threshold=1.0,
            ),
            "slice_span": _heuristic(
                label="Slice span",
                severity=HEURISTIC_WARNING,
                comparison="greater_than_or_equal",
                threshold=2.0,
            ),
        },
        representative_total_ms_budget=90.0,
        worst_case_total_ms_budget=190.0,
        final_release_allowed=True,
        portable_bundle_allowed=True,
        transparency_artifact_allowed=True,
        requires_operator_review_on_warning=True,
        success_decision_class=DECISION_CLASS_FINAL_RELEASE,
        **_COMMON_TOLERANCES,
    ),
    "strict_research_mode": _profile(
        profile_name="strict_research_mode",
        sequence_family="generic_brain_mri",
        clinical_mode="strict_research_final_release",
        purpose="High-scrutiny reproducibility and audit review.",
        risk_posture="strict",
        operator_interpretation=(
            "Stricter operational posture for audit and reproducibility; not inherently more clinically correct."
        ),
        icsd_policy_id="policy/mri-guardian/strict-research",
        threshold=2.5,
        guard_band_fraction=0.9,
        required_authority_source_types=_COMMON_FINAL_AUTHORITIES + ("research_protocol",),
        heuristic_rules={
            "mask_fraction_guardrail": _heuristic(
                label="Verified occupancy fraction",
                severity=HEURISTIC_BLOCKING,
                comparison="less_than_or_equal",
                threshold=0.30,
            ),
            "inside_outside_contrast": _heuristic(
                label="Inside/outside contrast delta",
                severity=HEURISTIC_BLOCKING,
                comparison="greater_than_or_equal",
                threshold=0.90,
            ),
            "slice_span": _heuristic(
                label="Slice span",
                severity=HEURISTIC_BLOCKING,
                comparison="greater_than_or_equal",
                threshold=3.0,
            ),
        },
        representative_total_ms_budget=95.0,
        worst_case_total_ms_budget=200.0,
        final_release_allowed=True,
        portable_bundle_allowed=True,
        transparency_artifact_allowed=True,
        requires_operator_review_on_warning=True,
        success_decision_class=DECISION_CLASS_FINAL_RELEASE,
        spatial_per_voxel_abs_error=_COMMON_TOLERANCES["spatial_per_voxel_abs_error"] / 10.0,
        spatial_global_relative_l2_error=_COMMON_TOLERANCES["spatial_global_relative_l2_error"] / 10.0,
        frequency_per_coefficient_abs_error=_COMMON_TOLERANCES["frequency_per_coefficient_abs_error"] / 10.0,
        frequency_global_relative_l2_error=_COMMON_TOLERANCES["frequency_global_relative_l2_error"] / 10.0,
        imaginary_leakage_abs_error=_COMMON_TOLERANCES["imaginary_leakage_abs_error"] / 10.0,
        volume_absolute_error=_COMMON_TOLERANCES["volume_absolute_error"] / 10.0,
    ),
    "fast_triage_mode": _profile(
        profile_name="fast_triage_mode",
        sequence_family="generic_brain_mri",
        clinical_mode="triage_non_final",
        purpose="Non-final early screening with explicit semantic guardrails.",
        risk_posture="triage",
        operator_interpretation="Triage-only output must not be treated as a final verified release.",
        icsd_policy_id="policy/mri-guardian/fast-triage",
        threshold=2.5,
        guard_band_fraction=0.7,
        required_authority_source_types=(
            "scanner_manifest",
            "voxel_geometry",
            "model_manifest",
        ),
        heuristic_rules={
            "mask_fraction_guardrail": _heuristic(
                label="Verified occupancy fraction",
                severity=HEURISTIC_ADVISORY,
                comparison="less_than_or_equal",
                threshold=0.40,
            ),
            "inside_outside_contrast": _heuristic(
                label="Inside/outside contrast delta",
                severity=HEURISTIC_DISABLED,
                comparison="greater_than_or_equal",
                threshold=0.0,
                enabled=False,
            ),
            "slice_span": _heuristic(
                label="Slice span",
                severity=HEURISTIC_DISABLED,
                comparison="greater_than_or_equal",
                threshold=0.0,
                enabled=False,
            ),
        },
        representative_total_ms_budget=60.0,
        worst_case_total_ms_budget=140.0,
        final_release_allowed=False,
        portable_bundle_allowed=False,
        transparency_artifact_allowed=False,
        requires_operator_review_on_warning=False,
        success_decision_class=DECISION_CLASS_TRIAGE_ONLY,
        spatial_per_voxel_abs_error=_COMMON_TOLERANCES["spatial_per_voxel_abs_error"] * 5.0,
        spatial_global_relative_l2_error=_COMMON_TOLERANCES["spatial_global_relative_l2_error"] * 5.0,
        frequency_per_coefficient_abs_error=_COMMON_TOLERANCES["frequency_per_coefficient_abs_error"] * 5.0,
        frequency_global_relative_l2_error=_COMMON_TOLERANCES["frequency_global_relative_l2_error"] * 5.0,
        imaginary_leakage_abs_error=_COMMON_TOLERANCES["imaginary_leakage_abs_error"] * 5.0,
        volume_absolute_error=_COMMON_TOLERANCES["volume_absolute_error"] * 5.0,
    ),
}
