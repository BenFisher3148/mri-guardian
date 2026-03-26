"""Canonical deterministic mask-engine catalogue and dispatcher.

This module is intentionally rigid. Algorithm identifiers are proof-significant
names and must remain stable once published in released profiles or replay
artifacts.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

CANONICAL_MASK_INPUT_SEMANTICS = "raw_derived_reconstruction"
CANONICAL_MASK_OUTPUT_SEMANTICS = "verified_mask"

THRESHOLD_ALGORITHM = "threshold_on_raw_derived_reconstruction"
MORPHOLOGY_ALGORITHM = "morphology_constrained_threshold_on_raw_derived_reconstruction"
CONNECTED_COMPONENT_ALGORITHM = (
    "connected_component_constrained_threshold_on_raw_derived_reconstruction"
)
SLICE_CONSISTENCY_ALGORITHM = (
    "slice_consistency_constrained_threshold_on_raw_derived_reconstruction"
)

_CANONICAL_COMPARISONS = ("greater_than_or_equal",)
_CANONICAL_CONNECTIVITIES = ("faces",)
_CANONICAL_SELECTION_RULES = ("keep_all_components_meeting_min_size",)
_CANONICAL_CONSISTENCY_UNITS = ("in_slice_component",)
_CANONICAL_IN_SLICE_CONNECTIVITIES = ("faces_2d",)
_CANONICAL_OVERLAP_RULES = ("any_overlap",)
_ALGORITHM_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")
_FACES_NEIGHBOR_OFFSETS = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)
_FACES_2D_NEIGHBOR_OFFSETS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)
_REQUIRED = object()


@dataclass(frozen=True)
class ParameterSpec:
    """Canonical parameter normalisation rule for one approved algorithm field."""

    name: str
    kind: str
    default: Any = _REQUIRED
    allowed_values: tuple[str, ...] = ()
    minimum: int | float | None = None


@dataclass(frozen=True)
class ResolvedMaskPolicy:
    """Proof-significant resolved mask policy."""

    algorithm: str
    parameters: dict[str, Any]
    deterministic: bool = True
    input_semantics: str = CANONICAL_MASK_INPUT_SEMANTICS
    output_semantics: str = CANONICAL_MASK_OUTPUT_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "deterministic": self.deterministic,
            "input_semantics": self.input_semantics,
            "output_semantics": self.output_semantics,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class MaskEngineRuntimeMetadata:
    """Non-canonical runtime metadata that must never enter proof artifacts."""

    operation_counts: dict[str, int]
    debug_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaskEngineExecution:
    """One canonical mask-engine execution result."""

    verified_mask: np.ndarray
    policy: ResolvedMaskPolicy
    runtime_metadata: MaskEngineRuntimeMetadata


@dataclass(frozen=True)
class _MaskAlgorithmSpec:
    algorithm: str
    parameter_specs: tuple[ParameterSpec, ...]
    implementation: Callable[[np.ndarray, ResolvedMaskPolicy], MaskEngineExecution]


@dataclass(frozen=True)
class _SliceComponentNode:
    node_id: int
    slice_index: int
    footprint: frozenset[tuple[int, int]]


def list_approved_mask_algorithms() -> tuple[str, ...]:
    """Return the fixed in-repo catalogue of approved mask algorithms."""
    return tuple(_APPROVED_ALGORITHM_TABLE)


def resolve_mask_policy(mask_rules: Mapping[str, Any]) -> ResolvedMaskPolicy:
    """Resolve one mask policy into its canonical proof-significant form."""
    if not isinstance(mask_rules, Mapping):
        raise TypeError("mask_rules must be a mapping.")
    unknown_top_level = set(mask_rules) - {
        "algorithm",
        "deterministic",
        "input_semantics",
        "output_semantics",
        "parameters",
    }
    if unknown_top_level:
        names = ", ".join(sorted(str(key) for key in unknown_top_level))
        raise ValueError(f"mask_rules contains unsupported top-level keys: {names}.")
    raw_algorithm = _normalise_identifier(mask_rules.get("algorithm"), field_name="algorithm")
    try:
        algorithm_spec = _APPROVED_ALGORITHM_TABLE[raw_algorithm]
    except KeyError as exc:
        raise ValueError(f"Unsupported approved mask algorithm: {raw_algorithm!r}.") from exc
    deterministic = _normalise_bool(
        mask_rules.get("deterministic", True),
        field_name="deterministic",
    )
    if not deterministic:
        raise ValueError("Mask derivation policies must remain deterministic.")
    input_semantics = _normalise_identifier(
        mask_rules.get("input_semantics", CANONICAL_MASK_INPUT_SEMANTICS),
        field_name="input_semantics",
    )
    if input_semantics != CANONICAL_MASK_INPUT_SEMANTICS:
        raise ValueError(
            "Unsupported mask input_semantics: "
            f"{input_semantics!r}."
        )
    output_semantics = _normalise_identifier(
        mask_rules.get("output_semantics", CANONICAL_MASK_OUTPUT_SEMANTICS),
        field_name="output_semantics",
    )
    if output_semantics != CANONICAL_MASK_OUTPUT_SEMANTICS:
        raise ValueError(
            "Unsupported mask output_semantics: "
            f"{output_semantics!r}."
        )
    raw_parameters = mask_rules.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise TypeError("mask_rules['parameters'] must be a mapping.")
    unknown_parameters = set(raw_parameters) - {
        parameter.name for parameter in algorithm_spec.parameter_specs
    }
    if unknown_parameters:
        names = ", ".join(sorted(str(key) for key in unknown_parameters))
        raise ValueError(
            f"mask_rules contains parameters that are not applicable to {raw_algorithm!r}: "
            f"{names}."
        )
    parameters: dict[str, Any] = {}
    for parameter_spec in algorithm_spec.parameter_specs:
        if parameter_spec.name in raw_parameters:
            raw_value = raw_parameters[parameter_spec.name]
        elif parameter_spec.default is not _REQUIRED:
            raw_value = parameter_spec.default
        else:
            raise ValueError(
                f"mask_rules for {raw_algorithm!r} is missing required parameter "
                f"{parameter_spec.name!r}."
            )
        parameters[parameter_spec.name] = _normalise_parameter(
            raw_value,
            parameter_spec=parameter_spec,
            algorithm=raw_algorithm,
        )
    return ResolvedMaskPolicy(
        algorithm=raw_algorithm,
        parameters=parameters,
        deterministic=deterministic,
        input_semantics=input_semantics,
        output_semantics=output_semantics,
    )


def derive_verified_mask(
    derived_reconstruction: np.ndarray,
    mask_rules: Mapping[str, Any],
) -> MaskEngineExecution:
    """Execute the canonical approved algorithm for one resolved mask policy."""
    policy = resolve_mask_policy(mask_rules)
    reconstruction = np.asarray(derived_reconstruction, dtype=np.float64)
    if reconstruction.ndim != 3:
        raise ValueError(
            "Mask derivation expects a 3D float64 reconstruction on the measured grid."
        )
    return _APPROVED_ALGORITHM_TABLE[policy.algorithm].implementation(reconstruction, policy)


def _threshold_execution(
    reconstruction: np.ndarray,
    policy: ResolvedMaskPolicy,
) -> MaskEngineExecution:
    seed_mask = _threshold_seed_mask(reconstruction, policy.parameters)
    return MaskEngineExecution(
        verified_mask=seed_mask.astype(np.int64),
        policy=policy,
        runtime_metadata=MaskEngineRuntimeMetadata(
            operation_counts={
                "threshold_comparison_passes": 1,
                "voxel_count": int(reconstruction.size),
            }
        ),
    )


def _morphology_execution(
    reconstruction: np.ndarray,
    policy: ResolvedMaskPolicy,
) -> MaskEngineExecution:
    # This engine is one canonical procedure: threshold seed generation followed
    # by boolean opening and boolean closing with Manhattan-radius structuring
    # elements under conceptual zero-padding.
    seed_mask = _threshold_seed_mask(reconstruction, policy.parameters)
    opening_radius = int(policy.parameters["opening_radius_voxels"])
    closing_radius = int(policy.parameters["closing_radius_voxels"])
    opened_mask = _binary_dilation(_binary_erosion(seed_mask, opening_radius), opening_radius)
    closed_mask = _binary_erosion(_binary_dilation(opened_mask, closing_radius), closing_radius)
    return MaskEngineExecution(
        verified_mask=closed_mask.astype(np.int64),
        policy=policy,
        runtime_metadata=MaskEngineRuntimeMetadata(
            operation_counts={
                "threshold_comparison_passes": 1,
                "opening_structuring_element_size": len(
                    _manhattan_offsets(opening_radius)
                ),
                "closing_structuring_element_size": len(
                    _manhattan_offsets(closing_radius)
                ),
                "boolean_morphology_steps": 4,
            }
        ),
    )


def _connected_component_execution(
    reconstruction: np.ndarray,
    policy: ResolvedMaskPolicy,
) -> MaskEngineExecution:
    seed_mask = _threshold_seed_mask(reconstruction, policy.parameters)
    labels, component_sizes = _label_connected_components(
        seed_mask,
        connectivity=str(policy.parameters["connectivity"]),
    )
    selection_rule = str(policy.parameters["selection_rule"])
    if selection_rule != "keep_all_components_meeting_min_size":
        raise ValueError(f"Unsupported component selection rule: {selection_rule!r}.")
    min_component_voxels = int(policy.parameters["min_component_voxels"])
    retained_labels = tuple(
        label
        for label, size in component_sizes.items()
        if size >= min_component_voxels
    )
    retained_mask = np.zeros(seed_mask.shape, dtype=bool)
    for label in retained_labels:
        retained_mask |= labels == label
    return MaskEngineExecution(
        verified_mask=retained_mask.astype(np.int64),
        policy=policy,
        runtime_metadata=MaskEngineRuntimeMetadata(
            operation_counts={
                "threshold_comparison_passes": 1,
                "component_count": len(component_sizes),
                "retained_component_count": len(retained_labels),
                "rejected_component_count": len(component_sizes) - len(retained_labels),
            },
            debug_notes=(
                "component_labels_are_internal_execution_state_only",
                "component_traversal_uses_queue_based_bfs",
            ),
        ),
    )


def _slice_consistency_execution(
    reconstruction: np.ndarray,
    policy: ResolvedMaskPolicy,
) -> MaskEngineExecution:
    seed_mask = _threshold_seed_mask(reconstruction, policy.parameters)
    consistency_unit = str(policy.parameters["consistency_unit"])
    if consistency_unit != "in_slice_component":
        raise ValueError(f"Unsupported slice consistency unit: {consistency_unit!r}.")
    nodes_by_slice = _label_in_slice_components(
        seed_mask,
        connectivity=str(policy.parameters["in_slice_connectivity"]),
    )
    overlap_rule = str(policy.parameters["overlap_rule"])
    if overlap_rule != "any_overlap":
        raise ValueError(f"Unsupported slice overlap rule: {overlap_rule!r}.")
    adjacency = _build_slice_overlap_graph(nodes_by_slice, overlap_rule=overlap_rule)
    max_run_ending_at = _build_forward_run_lengths(nodes_by_slice, adjacency)
    max_run_starting_at = _build_reverse_run_lengths(nodes_by_slice, adjacency)
    min_consecutive_slices = int(policy.parameters["min_consecutive_slices"])
    retained_node_ids = {
        node.node_id
        for slice_nodes in nodes_by_slice
        for node in slice_nodes
        if (
            max_run_ending_at[node.node_id]
            + max_run_starting_at[node.node_id]
            - 1
            >= min_consecutive_slices
        )
    }
    retained_mask = np.zeros(seed_mask.shape, dtype=bool)
    for slice_nodes in nodes_by_slice:
        for node in slice_nodes:
            if node.node_id not in retained_node_ids:
                continue
            for y_index, x_index in sorted(node.footprint):
                retained_mask[node.slice_index, y_index, x_index] = True
    slice_component_count = sum(len(slice_nodes) for slice_nodes in nodes_by_slice)
    retained_component_count = len(retained_node_ids)
    return MaskEngineExecution(
        verified_mask=retained_mask.astype(np.int64),
        policy=policy,
        runtime_metadata=MaskEngineRuntimeMetadata(
            operation_counts={
                "threshold_comparison_passes": 1,
                "slice_component_count": slice_component_count,
                "retained_component_count": retained_component_count,
                "rejected_component_count": slice_component_count - retained_component_count,
                "overlap_edge_count": sum(len(neighbors) for neighbors in adjacency.values()) // 2,
            },
            debug_notes=(
                "slice_component_labels_are_internal_execution_state_only",
                "overlap_graph_node_ids_are_internal_execution_state_only",
                "qualifying_run_ids_are_internal_execution_state_only",
                "slice_component_traversal_uses_queue_based_bfs",
                "slice_component_survival_is_defined_by_qualifying_run_membership_not_label_rank",
            ),
        ),
    )


def _threshold_seed_mask(
    reconstruction: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    comparison = str(parameters["comparison"])
    threshold = float(parameters["threshold"])
    if comparison != "greater_than_or_equal":
        raise ValueError(f"Unsupported threshold comparison: {comparison!r}.")
    return reconstruction >= threshold


def _label_connected_components(
    seed_mask: np.ndarray,
    *,
    connectivity: str,
) -> tuple[np.ndarray, dict[int, int]]:
    if connectivity != "faces":
        raise ValueError(f"Unsupported connected-component connectivity: {connectivity!r}.")
    seed = np.asarray(seed_mask, dtype=bool)
    labels = np.zeros(seed.shape, dtype=np.int32)
    component_sizes: dict[int, int] = {}
    next_label = 1
    z_size, y_size, x_size = seed.shape
    for z_index in range(z_size):
        for y_index in range(y_size):
            for x_index in range(x_size):
                if not seed[z_index, y_index, x_index] or labels[z_index, y_index, x_index] != 0:
                    continue
                # Component labels are deterministic internal state only. Traversal
                # order is fixed now so future ranking-based families cannot drift.
                queue: deque[tuple[int, int, int]] = deque([(z_index, y_index, x_index)])
                labels[z_index, y_index, x_index] = next_label
                component_size = 0
                while queue:
                    current_z, current_y, current_x = queue.popleft()
                    component_size += 1
                    for delta_z, delta_y, delta_x in _FACES_NEIGHBOR_OFFSETS:
                        neighbor_z = current_z + delta_z
                        neighbor_y = current_y + delta_y
                        neighbor_x = current_x + delta_x
                        if not (
                            0 <= neighbor_z < z_size
                            and 0 <= neighbor_y < y_size
                            and 0 <= neighbor_x < x_size
                        ):
                            continue
                        if not seed[neighbor_z, neighbor_y, neighbor_x]:
                            continue
                        if labels[neighbor_z, neighbor_y, neighbor_x] != 0:
                            continue
                        labels[neighbor_z, neighbor_y, neighbor_x] = next_label
                        queue.append((neighbor_z, neighbor_y, neighbor_x))
                component_sizes[next_label] = component_size
                next_label += 1
    return labels, component_sizes


def _label_in_slice_components(
    seed_mask: np.ndarray,
    *,
    connectivity: str,
) -> tuple[tuple[_SliceComponentNode, ...], ...]:
    if connectivity != "faces_2d":
        raise ValueError(f"Unsupported in-slice connectivity: {connectivity!r}.")
    seed = np.asarray(seed_mask, dtype=bool)
    z_size, y_size, x_size = seed.shape
    next_node_id = 1
    nodes_by_slice: list[tuple[_SliceComponentNode, ...]] = []
    for z_index in range(z_size):
        slice_seed = seed[z_index]
        slice_labels = np.zeros(slice_seed.shape, dtype=np.int32)
        slice_nodes: list[_SliceComponentNode] = []
        for y_index in range(y_size):
            for x_index in range(x_size):
                if not slice_seed[y_index, x_index] or slice_labels[y_index, x_index] != 0:
                    continue
                # Traversal order is fixed for deterministic labelling and
                # reproducibility only; retention is governed by qualifying-run
                # membership rather than label rank.
                queue: deque[tuple[int, int]] = deque([(y_index, x_index)])
                slice_labels[y_index, x_index] = next_node_id
                footprint: list[tuple[int, int]] = []
                while queue:
                    current_y, current_x = queue.popleft()
                    footprint.append((current_y, current_x))
                    for delta_y, delta_x in _FACES_2D_NEIGHBOR_OFFSETS:
                        neighbor_y = current_y + delta_y
                        neighbor_x = current_x + delta_x
                        if not (0 <= neighbor_y < y_size and 0 <= neighbor_x < x_size):
                            continue
                        if not slice_seed[neighbor_y, neighbor_x]:
                            continue
                        if slice_labels[neighbor_y, neighbor_x] != 0:
                            continue
                        slice_labels[neighbor_y, neighbor_x] = next_node_id
                        queue.append((neighbor_y, neighbor_x))
                slice_nodes.append(
                    _SliceComponentNode(
                        node_id=next_node_id,
                        slice_index=z_index,
                        footprint=frozenset(footprint),
                    )
                )
                next_node_id += 1
        nodes_by_slice.append(tuple(slice_nodes))
    return tuple(nodes_by_slice)


def _build_slice_overlap_graph(
    nodes_by_slice: tuple[tuple[_SliceComponentNode, ...], ...],
    *,
    overlap_rule: str,
) -> dict[int, set[int]]:
    if overlap_rule != "any_overlap":
        raise ValueError(f"Unsupported slice overlap rule: {overlap_rule!r}.")
    adjacency = {
        node.node_id: set()
        for slice_nodes in nodes_by_slice
        for node in slice_nodes
    }
    for z_index in range(len(nodes_by_slice) - 1):
        current_nodes = nodes_by_slice[z_index]
        next_nodes = nodes_by_slice[z_index + 1]
        for current_node in current_nodes:
            for next_node in next_nodes:
                if current_node.footprint.isdisjoint(next_node.footprint):
                    continue
                adjacency[current_node.node_id].add(next_node.node_id)
                adjacency[next_node.node_id].add(current_node.node_id)
    return adjacency


def _build_forward_run_lengths(
    nodes_by_slice: tuple[tuple[_SliceComponentNode, ...], ...],
    adjacency: Mapping[int, set[int]],
) -> dict[int, int]:
    node_to_slice = _build_node_to_slice_index(nodes_by_slice)
    max_run_ending_at: dict[int, int] = {}
    for z_index, slice_nodes in enumerate(nodes_by_slice):
        for node in slice_nodes:
            predecessor_lengths = [
                max_run_ending_at[neighbor_id]
                for neighbor_id in adjacency[node.node_id]
                if node_to_slice[neighbor_id] == z_index - 1
            ]
            max_run_ending_at[node.node_id] = (
                1 + max(predecessor_lengths) if predecessor_lengths else 1
            )
    return max_run_ending_at


def _build_reverse_run_lengths(
    nodes_by_slice: tuple[tuple[_SliceComponentNode, ...], ...],
    adjacency: Mapping[int, set[int]],
) -> dict[int, int]:
    node_to_slice = _build_node_to_slice_index(nodes_by_slice)
    max_run_starting_at: dict[int, int] = {}
    for z_index in range(len(nodes_by_slice) - 1, -1, -1):
        for node in nodes_by_slice[z_index]:
            successor_lengths = [
                max_run_starting_at[neighbor_id]
                for neighbor_id in adjacency[node.node_id]
                if node_to_slice[neighbor_id] == z_index + 1
            ]
            max_run_starting_at[node.node_id] = (
                1 + max(successor_lengths) if successor_lengths else 1
            )
    return max_run_starting_at


def _build_node_to_slice_index(
    nodes_by_slice: tuple[tuple[_SliceComponentNode, ...], ...],
) -> dict[int, int]:
    return {
        node.node_id: node.slice_index
        for slice_nodes in nodes_by_slice
        for node in slice_nodes
    }


def _binary_erosion(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return np.asarray(mask, dtype=bool)
    padded = np.pad(mask.astype(bool), pad_width=((radius, radius),) * 3, constant_values=False)
    result = np.ones(mask.shape, dtype=bool)
    z_size, y_size, x_size = mask.shape
    for dz, dy, dx in _manhattan_offsets(radius):
        window = padded[
            radius + dz : radius + dz + z_size,
            radius + dy : radius + dy + y_size,
            radius + dx : radius + dx + x_size,
        ]
        result &= window
    return result


def _binary_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return np.asarray(mask, dtype=bool)
    padded = np.pad(mask.astype(bool), pad_width=((radius, radius),) * 3, constant_values=False)
    result = np.zeros(mask.shape, dtype=bool)
    z_size, y_size, x_size = mask.shape
    for dz, dy, dx in _manhattan_offsets(radius):
        window = padded[
            radius + dz : radius + dz + z_size,
            radius + dy : radius + dy + y_size,
            radius + dx : radius + dx + x_size,
        ]
        result |= window
    return result


def _manhattan_offsets(radius: int) -> tuple[tuple[int, int, int], ...]:
    if radius < 0:
        raise ValueError(f"Structuring-element radius must be non-negative, got {radius!r}.")
    return tuple(
        sorted(
            (
                (dz, dy, dx)
                for dz in range(-radius, radius + 1)
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
                if abs(dz) + abs(dy) + abs(dx) <= radius
            )
        )
    )


def _normalise_parameter(
    value: Any,
    *,
    parameter_spec: ParameterSpec,
    algorithm: str,
) -> Any:
    field_name = f"{algorithm}.{parameter_spec.name}"
    if parameter_spec.kind == "enum":
        normalised = _normalise_identifier(value, field_name=field_name)
        if normalised not in parameter_spec.allowed_values:
            allowed = ", ".join(repr(item) for item in parameter_spec.allowed_values)
            raise ValueError(
                f"{field_name} must be one of {allowed}; got {normalised!r}."
            )
        return normalised
    if parameter_spec.kind == "int":
        normalised = _normalise_int(value, field_name=field_name)
        if parameter_spec.minimum is not None and normalised < int(parameter_spec.minimum):
            raise ValueError(
                f"{field_name} must be >= {int(parameter_spec.minimum)!r}; got {normalised!r}."
            )
        return normalised
    if parameter_spec.kind == "float":
        normalised = _normalise_float(value, field_name=field_name)
        if parameter_spec.minimum is not None and normalised < float(parameter_spec.minimum):
            raise ValueError(
                f"{field_name} must be >= {float(parameter_spec.minimum)!r}; "
                f"got {normalised!r}."
            )
        return normalised
    if parameter_spec.kind == "bool":
        return _normalise_bool(value, field_name=field_name)
    raise ValueError(f"Unsupported parameter kind {parameter_spec.kind!r}.")


def _normalise_identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalised = value.strip().lower()
    if not normalised:
        raise ValueError(f"{field_name} must not be empty.")
    if any(character not in _ALGORITHM_NAME_CHARS for character in normalised):
        raise ValueError(
            f"{field_name} must use canonical lowercase identifier characters only."
        )
    return normalised


def _normalise_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, np.bool_):
        return bool(value)
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


def _normalise_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not a boolean.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer.")
    return int(value)


def _normalise_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be a finite float, not a boolean.")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{field_name} must be numeric.")
    normalised = float(value)
    if not np.isfinite(normalised):
        raise ValueError(f"{field_name} must be finite.")
    return normalised


def _validate_catalogue(specs: tuple[_MaskAlgorithmSpec, ...]) -> dict[str, _MaskAlgorithmSpec]:
    if not specs:
        raise ValueError("The approved mask algorithm catalogue must not be empty.")
    approved: dict[str, _MaskAlgorithmSpec] = {}
    for spec in specs:
        if spec.algorithm in approved:
            raise ValueError(f"Duplicate approved mask algorithm name: {spec.algorithm!r}.")
        _normalise_identifier(spec.algorithm, field_name="algorithm")
        if not callable(spec.implementation):
            raise ValueError(
                f"Approved mask algorithm {spec.algorithm!r} must define an implementation."
            )
        if not spec.parameter_specs:
            raise ValueError(
                f"Approved mask algorithm {spec.algorithm!r} must define parameter specs."
            )
        parameter_names: set[str] = set()
        for parameter_spec in spec.parameter_specs:
            _normalise_identifier(parameter_spec.name, field_name="parameter")
            if parameter_spec.name in parameter_names:
                raise ValueError(
                    f"Approved mask algorithm {spec.algorithm!r} contains duplicate "
                    f"parameter {parameter_spec.name!r}."
                )
            parameter_names.add(parameter_spec.name)
        approved[spec.algorithm] = spec
    return approved


_APPROVED_ALGORITHM_TABLE = _validate_catalogue(
    (
        _MaskAlgorithmSpec(
            algorithm=THRESHOLD_ALGORITHM,
            parameter_specs=(
                ParameterSpec(
                    name="comparison",
                    kind="enum",
                    default="greater_than_or_equal",
                    allowed_values=_CANONICAL_COMPARISONS,
                ),
                ParameterSpec(name="threshold", kind="float"),
            ),
            implementation=_threshold_execution,
        ),
        _MaskAlgorithmSpec(
            algorithm=MORPHOLOGY_ALGORITHM,
            parameter_specs=(
                ParameterSpec(
                    name="comparison",
                    kind="enum",
                    default="greater_than_or_equal",
                    allowed_values=_CANONICAL_COMPARISONS,
                ),
                ParameterSpec(name="threshold", kind="float"),
                ParameterSpec(
                    name="opening_radius_voxels",
                    kind="int",
                    default=1,
                    minimum=0,
                ),
                ParameterSpec(
                    name="closing_radius_voxels",
                    kind="int",
                    default=1,
                    minimum=0,
                ),
            ),
            implementation=_morphology_execution,
        ),
        _MaskAlgorithmSpec(
            algorithm=CONNECTED_COMPONENT_ALGORITHM,
            parameter_specs=(
                ParameterSpec(
                    name="comparison",
                    kind="enum",
                    default="greater_than_or_equal",
                    allowed_values=_CANONICAL_COMPARISONS,
                ),
                ParameterSpec(name="threshold", kind="float"),
                ParameterSpec(
                    name="connectivity",
                    kind="enum",
                    default="faces",
                    allowed_values=_CANONICAL_CONNECTIVITIES,
                ),
                ParameterSpec(
                    name="min_component_voxels",
                    kind="int",
                    minimum=1,
                ),
                ParameterSpec(
                    name="selection_rule",
                    kind="enum",
                    default="keep_all_components_meeting_min_size",
                    allowed_values=_CANONICAL_SELECTION_RULES,
                ),
            ),
            implementation=_connected_component_execution,
        ),
        _MaskAlgorithmSpec(
            algorithm=SLICE_CONSISTENCY_ALGORITHM,
            parameter_specs=(
                ParameterSpec(
                    name="comparison",
                    kind="enum",
                    default="greater_than_or_equal",
                    allowed_values=_CANONICAL_COMPARISONS,
                ),
                ParameterSpec(name="threshold", kind="float"),
                ParameterSpec(
                    name="consistency_unit",
                    kind="enum",
                    default="in_slice_component",
                    allowed_values=_CANONICAL_CONSISTENCY_UNITS,
                ),
                ParameterSpec(
                    name="in_slice_connectivity",
                    kind="enum",
                    default="faces_2d",
                    allowed_values=_CANONICAL_IN_SLICE_CONNECTIVITIES,
                ),
                ParameterSpec(
                    name="min_consecutive_slices",
                    kind="int",
                    minimum=2,
                ),
                ParameterSpec(
                    name="overlap_rule",
                    kind="enum",
                    default="any_overlap",
                    allowed_values=_CANONICAL_OVERLAP_RULES,
                ),
            ),
            implementation=_slice_consistency_execution,
        ),
    )
)
