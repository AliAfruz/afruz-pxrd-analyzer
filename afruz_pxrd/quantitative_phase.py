from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .phase_identification import ReferenceEntry, prepare_reference_peaks


QPA_MODES = (
    "RIR-corrected weight fractions",
    "Pattern scale fractions",
)

QPA_WEIGHTING = (
    "Uniform",
    "Poisson-like",
    "Balanced",
)


@dataclass
class QPAPhaseSpec:
    reference: ReferenceEntry
    zero_shift_deg: float = 0.0
    rir_override: float | None = None


def _pseudo_voigt_height(
    x: np.ndarray,
    center: float,
    fwhm: float,
    eta: float,
) -> np.ndarray:
    width = max(float(fwhm), np.finfo(float).eps)
    delta = (np.asarray(x, dtype=float) - float(center)) / width
    gaussian = np.exp(-4.0 * np.log(2.0) * delta * delta)
    lorentzian = 1.0 / (1.0 + 4.0 * delta * delta)
    mixing = float(np.clip(eta, 0.0, 1.0))
    return mixing * lorentzian + (1.0 - mixing) * gaussian


def build_reference_basis(
    x: np.ndarray,
    peaks: list[dict],
    fwhm_deg: float,
    eta: float = 0.5,
    zero_shift_deg: float = 0.0,
) -> np.ndarray:
    """
    Build a continuous reference pattern from relative-intensity sticks.

    Peak heights retain the reference-card convention where the strongest
    reflection has height 1.0. This makes the fitted scale coefficient broadly
    compatible with the strongest-line RIR convention, but it remains a
    semi-quantitative approximation rather than a Rietveld scale factor.
    """
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    fwhm = max(float(fwhm_deg), np.median(np.diff(x)) * 1.05)
    half_window = 10.0 * fwhm

    for peak in peaks:
        center = float(peak["two_theta"]) + float(zero_shift_deg)
        intensity = max(0.0, float(peak.get("intensity", 0.0))) / 100.0
        if intensity <= 0:
            continue
        left = int(np.searchsorted(x, center - half_window, side="left"))
        right = int(np.searchsorted(x, center + half_window, side="right"))
        if right <= left:
            continue
        result[left:right] += intensity * _pseudo_voigt_height(
            x[left:right], center, fwhm, eta
        )

    return result


def _weight_vector(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "Uniform":
        return np.ones_like(y)

    shifted = y - float(np.min(y))
    positive = shifted[shifted > 0]
    floor = (
        float(np.percentile(positive, 10.0))
        if positive.size
        else 1.0
    )
    floor = max(floor, np.finfo(float).eps)

    if mode == "Poisson-like":
        weights = 1.0 / np.sqrt(np.maximum(shifted, floor))
    elif mode == "Balanced":
        scale = max(float(np.percentile(shifted, 95.0)), floor)
        weights = 1.0 / np.sqrt(0.10 + shifted / scale)
    else:
        raise ValueError(f"Unsupported QPA weighting mode: {mode}")

    mean = float(np.mean(weights))
    return weights / mean if mean > 0 else np.ones_like(y)


def _design_matrix(
    x: np.ndarray,
    phase_columns: list[np.ndarray],
    baseline_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    center = float(np.mean(x))
    half_span = max(float(np.ptp(x)) / 2.0, np.finfo(float).eps)
    xn = (x - center) / half_span

    columns = list(phase_columns)
    for degree in range(int(baseline_order) + 1):
        columns.append(xn ** degree)
    return np.column_stack(columns), xn


def _solve_coefficients(
    design: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    phase_count: int,
) -> np.ndarray:
    weighted_design = design * weights[:, None]
    weighted_y = y * weights
    lower = np.concatenate(
        [np.zeros(phase_count), np.full(design.shape[1] - phase_count, -np.inf)]
    )
    upper = np.full(design.shape[1], np.inf)
    result = lsq_linear(
        weighted_design,
        weighted_y,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=5000,
    )
    if not result.success:
        raise RuntimeError(f"QPA least-squares solution failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _fractions_from_scales(
    scales: np.ndarray,
    rir_values: np.ndarray,
    mode: str,
) -> np.ndarray:
    scales = np.maximum(0.0, np.asarray(scales, dtype=float))
    if mode == "RIR-corrected weight fractions":
        if np.any(~np.isfinite(rir_values)) or np.any(rir_values <= 0):
            raise ValueError(
                "Every quantified phase requires a positive RIR value in "
                "RIR-corrected mode. Use a Jade/PDF card containing I/Ic(RIR), "
                "enter an override, or use pattern scale fractions."
            )
        values = scales / rir_values
    elif mode == "Pattern scale fractions":
        values = scales
    else:
        raise ValueError(f"Unsupported QPA mode: {mode}")

    total = float(np.sum(values))
    if total <= np.finfo(float).eps:
        raise ValueError("All fitted phase scale coefficients are zero.")
    return values / total


def _internal_standard_correction(
    fractions: np.ndarray,
    standard_index: int | None,
    known_standard_percent: float | None,
) -> tuple[np.ndarray, float | None, list[str]]:
    fractions = np.asarray(fractions, dtype=float)
    warnings: list[str] = []
    if standard_index is None:
        return fractions * 100.0, None, warnings

    if known_standard_percent is None:
        raise ValueError("Known internal-standard wt% is required.")
    known_fraction = float(known_standard_percent) / 100.0
    if not 0.0 < known_fraction < 1.0:
        raise ValueError("Internal-standard wt% must lie between 0 and 100.")

    measured_standard = float(fractions[int(standard_index)])
    if measured_standard <= np.finfo(float).eps:
        raise ValueError("The fitted internal-standard fraction is zero.")

    crystalline_total = known_fraction / measured_standard
    corrected = fractions * crystalline_total * 100.0
    amorphous_percent = 100.0 * (1.0 - crystalline_total)
    if amorphous_percent < 0:
        warnings.append(
            "Internal-standard correction produced a negative amorphous fraction; "
            "verify RIR values, phase selection, peak overlap, and the added-standard amount."
        )
    if crystalline_total > 1.20:
        warnings.append(
            "Corrected crystalline total exceeds 120%; the internal-standard result is inconsistent."
        )
    return corrected, float(amorphous_percent), warnings


def quantify_phases(
    x: np.ndarray,
    y: np.ndarray,
    phase_specs: list[QPAPhaseSpec],
    target_wavelength_angstrom: float,
    mode: str = "RIR-corrected weight fractions",
    convert_from_d: bool = True,
    reference_intensity_cutoff_percent: float = 1.0,
    reference_fwhm_deg: float = 0.20,
    pseudo_voigt_eta: float = 0.5,
    baseline_order: int = 1,
    weighting: str = "Balanced",
    bootstrap_samples: int = 50,
    internal_standard_uid: str | None = None,
    known_internal_standard_percent: float | None = None,
    random_seed: int = 20260727,
) -> dict:
    """
    Semi-quantitative whole-pattern RIR analysis.

    The phase scale factors are obtained from a non-negative linear combination
    of broadened reference-card patterns plus a polynomial background. RIR
    correction uses scale_i / RIR_i followed by closure to 100%. This is not a
    Rietveld scale-factor method and must be reported as semi-quantitative.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("X and Y must be one-dimensional arrays of equal length.")
    if len(x) < 20:
        raise ValueError("At least twenty points are required for QPA.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("X values must be strictly increasing.")
    if not phase_specs:
        raise ValueError("At least one candidate phase is required for QPA.")
    if not 0 <= int(baseline_order) <= 4:
        raise ValueError("Baseline order must be between 0 and 4.")
    if mode not in QPA_MODES:
        raise ValueError(f"Unsupported QPA mode: {mode}")
    if weighting not in QPA_WEIGHTING:
        raise ValueError(f"Unsupported QPA weighting mode: {weighting}")

    intensity_scale = max(
        float(np.percentile(y, 99.5) - np.percentile(y, 1.0)),
        float(np.max(np.abs(y))) * 0.10,
        np.finfo(float).eps,
    )
    y_scaled = y / intensity_scale
    phase_columns: list[np.ndarray] = []
    phase_rows: list[dict] = []
    rir_values = []

    for spec in phase_specs:
        reference = spec.reference
        peaks = prepare_reference_peaks(
            reference,
            target_wavelength_angstrom=float(target_wavelength_angstrom),
            convert_from_d=bool(convert_from_d),
            intensity_cutoff_percent=float(reference_intensity_cutoff_percent),
            two_theta_min=float(x[0]),
            two_theta_max=float(x[-1]),
        )
        if len(peaks) < 2:
            raise ValueError(
                f"Reference {reference.name!r} has fewer than two usable peaks "
                "in the experimental range."
            )
        basis = build_reference_basis(
            x,
            peaks,
            fwhm_deg=float(reference_fwhm_deg),
            eta=float(pseudo_voigt_eta),
            zero_shift_deg=float(spec.zero_shift_deg),
        )
        if float(np.max(basis)) <= np.finfo(float).eps:
            raise ValueError(f"Reference {reference.name!r} produced an empty basis pattern.")
        phase_columns.append(basis)

        metadata_rir = reference.metadata.get("rir")
        rir = spec.rir_override if spec.rir_override is not None else metadata_rir
        try:
            rir_value = float(rir) if rir is not None else float("nan")
        except (TypeError, ValueError):
            rir_value = float("nan")
        rir_values.append(rir_value)
        phase_rows.append(
            {
                "reference_uid": reference.uid,
                "reference_name": reference.name,
                "formula": reference.formula,
                "source": reference.source,
                "zero_shift_deg": float(spec.zero_shift_deg),
                "rir": None if not np.isfinite(rir_value) else rir_value,
                "prepared_peak_count": len(peaks),
            }
        )

    design, _ = _design_matrix(x, phase_columns, int(baseline_order))
    weights = _weight_vector(y_scaled, weighting)
    coefficients = _solve_coefficients(
        design,
        y_scaled,
        weights,
        phase_count=len(phase_specs),
    )
    phase_scales = coefficients[: len(phase_specs)]
    calculated_scaled = design @ coefficients
    residual_scaled = y_scaled - calculated_scaled
    calculated = calculated_scaled * intensity_scale
    residual = y - calculated
    baseline_scaled = design[:, len(phase_specs):] @ coefficients[len(phase_specs):]
    baseline = baseline_scaled * intensity_scale
    contributions = [
        phase_scales[index] * phase_columns[index] * intensity_scale
        for index in range(len(phase_specs))
    ]

    rir_array = np.asarray(rir_values, dtype=float)
    fractions = _fractions_from_scales(phase_scales, rir_array, mode)

    standard_index = None
    if internal_standard_uid:
        matches = [
            index
            for index, row in enumerate(phase_rows)
            if row["reference_uid"] == internal_standard_uid
        ]
        if not matches:
            raise ValueError("The selected internal standard is not in the quantified phase set.")
        if mode != "RIR-corrected weight fractions":
            raise ValueError("Internal-standard correction requires RIR-corrected mode.")
        standard_index = matches[0]

    corrected_percent, amorphous_percent, warnings = _internal_standard_correction(
        fractions,
        standard_index,
        known_internal_standard_percent,
    )

    bootstrap_samples = int(np.clip(bootstrap_samples, 0, 1000))
    fraction_samples = []
    corrected_samples = []
    amorphous_samples = []
    if bootstrap_samples:
        rng = np.random.default_rng(int(random_seed))
        for _ in range(bootstrap_samples):
            sampled_residual = residual_scaled[
                rng.integers(0, len(residual_scaled), size=len(residual_scaled))
            ]
            boot_y = calculated_scaled + sampled_residual
            try:
                boot_coefficients = _solve_coefficients(
                    design,
                    boot_y,
                    weights,
                    phase_count=len(phase_specs),
                )
                boot_fractions = _fractions_from_scales(
                    boot_coefficients[: len(phase_specs)],
                    rir_array,
                    mode,
                )
                boot_corrected, boot_amorphous, _ = _internal_standard_correction(
                    boot_fractions,
                    standard_index,
                    known_internal_standard_percent,
                )
            except Exception:
                continue
            fraction_samples.append(boot_fractions * 100.0)
            corrected_samples.append(boot_corrected)
            if boot_amorphous is not None:
                amorphous_samples.append(boot_amorphous)

    fraction_uncertainty = np.full(len(phase_specs), np.nan)
    corrected_uncertainty = np.full(len(phase_specs), np.nan)
    if len(fraction_samples) >= 5:
        fraction_uncertainty = np.std(
            np.asarray(fraction_samples, dtype=float), axis=0, ddof=1
        )
        corrected_uncertainty = np.std(
            np.asarray(corrected_samples, dtype=float), axis=0, ddof=1
        )
    amorphous_uncertainty = (
        float(np.std(amorphous_samples, ddof=1))
        if len(amorphous_samples) >= 5
        else None
    )

    scale_total = max(float(np.sum(phase_scales)), np.finfo(float).eps)
    for index, row in enumerate(phase_rows):
        row.update(
            {
                "scale_coefficient": float(phase_scales[index]),
                "scale_fraction_percent": float(100.0 * phase_scales[index] / scale_total),
                "weight_fraction_percent": float(100.0 * fractions[index]),
                "weight_fraction_uncertainty_percent": (
                    None
                    if not np.isfinite(fraction_uncertainty[index])
                    else float(fraction_uncertainty[index])
                ),
                "corrected_weight_percent": float(corrected_percent[index]),
                "corrected_uncertainty_percent": (
                    None
                    if not np.isfinite(corrected_uncertainty[index])
                    else float(corrected_uncertainty[index])
                ),
            }
        )

    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean(residual * residual)))
    profile_residual = float(
        100.0 * np.sum(np.abs(residual)) / max(np.sum(np.abs(y)), np.finfo(float).eps)
    )
    weighted_profile_residual = float(
        100.0
        * math.sqrt(
            np.sum((weights * residual_scaled) ** 2)
            / max(np.sum((weights * y_scaled) ** 2), np.finfo(float).eps)
        )
    )

    phase_matrix = np.column_stack(phase_columns)
    condition_number = float(np.linalg.cond(phase_matrix))
    maximum_basis_correlation = None
    if len(phase_columns) > 1:
        correlation = np.corrcoef(phase_matrix, rowvar=False)
        off_diagonal = np.abs(correlation - np.eye(len(phase_columns)))
        maximum_basis_correlation = float(np.nanmax(off_diagonal))
        if maximum_basis_correlation > 0.98:
            warnings.append(
                "Reference patterns are highly correlated; individual phase fractions may be unstable."
            )
    if len(phase_specs) == 1:
        warnings.append(
            "Only one phase was quantified; closure forces its crystalline fraction to 100%."
        )
    if profile_residual > 20.0:
        warnings.append(
            "Large profile residual indicates missing phases, poor background treatment, or unsuitable reference broadening."
        )
    if mode == "Pattern scale fractions":
        warnings.append(
            "Pattern scale fractions are not weight fractions because no RIR correction was applied."
        )

    return {
        "mode": mode,
        "target_wavelength_angstrom": float(target_wavelength_angstrom),
        "convert_from_d": bool(convert_from_d),
        "reference_intensity_cutoff_percent": float(reference_intensity_cutoff_percent),
        "reference_fwhm_deg": float(reference_fwhm_deg),
        "pseudo_voigt_eta": float(pseudo_voigt_eta),
        "baseline_order": int(baseline_order),
        "weighting": weighting,
        "bootstrap_samples_requested": bootstrap_samples,
        "bootstrap_samples_successful": len(fraction_samples),
        "internal_standard_uid": internal_standard_uid,
        "known_internal_standard_percent": known_internal_standard_percent,
        "amorphous_percent": amorphous_percent,
        "amorphous_uncertainty_percent": amorphous_uncertainty,
        "phases": phase_rows,
        "diagnostics": {
            "r_squared": float(r_squared),
            "rmse": rmse,
            "profile_residual_percent": profile_residual,
            "weighted_profile_residual_percent": weighted_profile_residual,
            "condition_number": condition_number,
            "maximum_basis_correlation": maximum_basis_correlation,
            "point_count": len(x),
            "phase_count": len(phase_specs),
        },
        "warnings": warnings,
        "plot": {
            "x": x.tolist(),
            "observed": y.tolist(),
            "calculated": calculated.tolist(),
            "baseline": baseline.tolist(),
            "residual": residual.tolist(),
            "contributions": [
                {
                    "reference_uid": phase_rows[index]["reference_uid"],
                    "reference_name": phase_rows[index]["reference_name"],
                    "y": contribution.tolist(),
                }
                for index, contribution in enumerate(contributions)
            ],
        },
    }
