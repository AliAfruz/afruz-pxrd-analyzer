from __future__ import annotations

import numpy as np
import pytest

from afruz_pxrd.advanced_background import (
    BACKGROUND_METHODS,
    BackgroundError,
    BackgroundParameters,
    detect_background,
)
from afruz_pxrd.processing import (
    ProcessingParameters,
    create_manual_peak,
    detect_peaks,
    merge_detected_with_manual,
    process_pattern,
    smart_detect_peaks,
)
from afruz_pxrd.smart_smoothing import (
    SMOOTHING_METHODS,
    SmoothingParameters,
    smart_smooth_pattern,
)


@pytest.fixture(scope="module")
def noisy_pattern():
    rng = np.random.default_rng(20260811)
    x = np.linspace(10.0, 70.0, 1201)
    baseline = 120.0 + 0.35 * x + 8.0 * np.sin(x / 11.0)
    peaks = (
        900.0 * np.exp(-0.5 * ((x - 24.0) / 0.16) ** 2)
        + 1250.0 * np.exp(-0.5 * ((x - 42.5) / 0.22) ** 2)
        + 650.0 * np.exp(-0.5 * ((x - 58.0) / 0.28) ** 2)
    )
    y = baseline + peaks + rng.normal(0.0, 10.0, size=len(x))
    return x, y, baseline


def test_regular_and_smart_peak_detection_find_known_peaks(noisy_pattern):
    x, y, _ = noisy_pattern
    regular = detect_peaks(x, y, prominence_fraction=0.04, minimum_distance_points=8)
    smart, diagnostics = smart_detect_peaks(x, y, sensitivity="Balanced")

    for expected in (24.0, 42.5, 58.0):
        assert min(abs(row["position"] - expected) for row in regular) < 0.12
        assert min(abs(row["position"] - expected) for row in smart) < 0.12
    assert diagnostics["noise_sigma"] > 0


@pytest.mark.parametrize("method", BACKGROUND_METHODS)
def test_every_background_method_returns_finite_aligned_arrays(method, noisy_pattern):
    x, y, _ = noisy_pattern
    result = detect_background(
        x,
        y,
        BackgroundParameters(
            method=method,
            iterations=20,
            smoothness=55.0,
            peak_protection=True,
        ),
    )
    for key in ("background", "corrected", "protected_mask"):
        assert np.asarray(result[key]).shape == y.shape
    assert np.all(np.isfinite(result["background"]))
    assert np.all(np.isfinite(result["corrected"]))
    assert np.median(np.abs(result["background"] - y)) > 0


@pytest.mark.parametrize("method", SMOOTHING_METHODS)
def test_every_smoothing_method_is_finite_and_peak_safe(method, noisy_pattern):
    x, y, _ = noisy_pattern
    result = smart_smooth_pattern(
        x,
        y,
        SmoothingParameters(
            method=method,
            strength=40.0,
            peak_protection=True,
            peak_preservation=85.0,
        ),
    )
    smoothed = np.asarray(result["smoothed"], dtype=float)
    assert smoothed.shape == y.shape
    assert np.all(np.isfinite(smoothed))
    assert np.std(np.diff(smoothed)) <= np.std(np.diff(y)) * 1.10
    for center in (24.0, 42.5, 58.0):
        mask = np.abs(x - center) <= 0.25
        assert np.max(smoothed[mask]) >= 0.70 * np.max(y[mask])


def test_processing_normalization_keeps_auxiliary_display_units_aligned(noisy_pattern):
    x, y, _ = noisy_pattern
    processed, auxiliary = process_pattern(
        x,
        y,
        ProcessingParameters(
            subtract_background=True,
            background_method="Polynomial",
            smooth=True,
            smoothing_method="Savitzky–Golay",
            normalize=True,
        ),
    )
    assert np.max(np.abs(processed)) == pytest.approx(100.0)
    assert np.max(np.abs(auxiliary["smoothed"])) == pytest.approx(100.0)
    assert auxiliary["smoothing_source"].shape == processed.shape
    assert auxiliary["background_display"].shape == processed.shape
    assert np.max(np.abs(auxiliary["smoothing_source"])) < 150.0


def test_manual_peak_identity_survives_detected_peak_merge(noisy_pattern):
    x, y, _ = noisy_pattern
    manual = create_manual_peak(x, y, 24.07, peak_uuid="manual-stable-id")
    merged = merge_detected_with_manual(
        [manual],
        [{"position": 24.02, "intensity": 800.0, "prominence": 500.0, "fwhm": 0.2}],
        duplicate_tolerance_deg=0.10,
    )
    assert len(merged) == 1
    assert merged[0]["peak_uuid"] == "manual-stable-id"
    assert merged[0]["manual"] is True
    assert merged[0]["protected"] is True


def test_background_rejects_non_monotonic_coordinates(noisy_pattern):
    x, y, _ = noisy_pattern
    broken = x.copy()
    broken[100], broken[101] = broken[101], broken[100]
    with pytest.raises(BackgroundError, match="strictly increasing"):
        detect_background(broken, y, BackgroundParameters(method="Polynomial"))
