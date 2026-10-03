# Phase 9 — Instrument-profile physics and uncertainty propagation

Release: 22.7.0
Status: implemented

## Scientific boundary

Phase 9 makes the calibrated instrument response an explicit scientific input,
not just a set of starting values. New profiles use schema version 3 and retain
the covariance needed to carry calibration uncertainty into later analyses.
Legacy version-2 JSON profiles load through an in-memory migration; unavailable
legacy covariance is labelled unavailable rather than invented.

## Position model

For symmetric Bragg–Brentano geometry the observed standard position is modeled
as

```text
2θobs = 2θref + Z − (2s/R) cos θ + [1/(2μR)] sin(2θ)
```

where `Z` is the constant zero shift, `s` is specimen displacement, `R` is the
goniometer radius, and `1/μ` is the inverse linear absorption coefficient for
the thick-specimen transparency approximation. The angular correction is
calculated in radians and reported in degrees. Zero shift, displacement, and
transparency are stored separately with a joint covariance/correlation matrix.

Transparency fitting requires at least seven matched reflections and at least
45° of 2θ coverage. This is a deliberate identifiability guard: these position
terms can be strongly correlated over a narrow angular interval.

## Width and shape model

Gaussian width follows Caglioti:

```text
HG² = U tan²θ + V tanθ + W
```

The calibrated model comparison retains pseudo-Voigt, Thompson–Cox–Hastings
(TCH), and split-profile compatibility. Every candidate now records parameter
names, standard errors, covariance, correlation, and normal-matrix condition
number. TCH keeps Gaussian `U,V,W` and Lorentzian `X,Y` components explicit.

New axial-divergence metadata uses the GSAS-style Finger–Cox–Jephcoat geometry
ratio

```text
SH/L = (sample length + receiving-slit length) / goniometer diameter
```

Afruz applies a normalized, geometry-constrained reduced SH/L profile whose
low-angle tail vanishes when SH/L is zero and weakens with increasing angle.
It is clearly classified as a reduced model, not a complete
fundamental-parameters convolution. Old `axial_asymmetry` values remain readable
for project compatibility.

## Radiation configuration

The calibration UI distinguishes:

- Cu Kα1 monochromated;
- resolved Cu Kα1/Kα2 with explicit wavelengths and Kα2/Kα1 intensity ratio;
- custom monochromatic radiation; and
- a custom doublet.

The Cu defaults are Kα1 = 1.5405929 Å, Kα2 = 1.5444274 Å, and
I(Kα2)/I(Kα1) = 0.5. Spectral components persist in the profile instead of
being collapsed into an undocumented effective wavelength.

## Downstream uncertainty and validity

- Unit-cell refinement can subtract the calibrated position correction and
  combines peak-position error with propagated position-correction uncertainty.
- Size/strain correction propagates both measured-peak and instrument-width
  errors through Gaussian-quadrature or Lorentzian-linear correction.
- Pawley/Le Bail and Rietveld refinement use compatible calibration covariance
  as a Gaussian prior for shared zero/U/V/W/X/Y parameters.
- The calibrated minimum/maximum 2θ interval is enforced by default. A caller
  must explicitly opt into extrapolation, and the result records that choice.
- Results retain the instrument-profile fingerprint, covariance-use flag, prior
  parameter names, and effective prior rank.

## Acceptance evidence

`tests/test_phase9_instrument_physics.py` covers:

1. resolved Cu Kα spectral positions and intensity ratio;
2. synthetic recovery of zero shift, displacement, and transparency;
3. retained position and width covariance;
4. normalized, angle-dependent SH/L axial behavior;
5. strict calibration-range rejection;
6. instrument-width uncertainty propagation into size/strain correction;
7. instrument-position uncertainty propagation into unit-cell refinement;
8. construction of downstream Gaussian priors; and
9. version-2 to version-3 profile migration.

## Scientific references

- McCusker et al., “Rietveld refinement guidelines,” *Journal of Applied
  Crystallography* 32 (1999), including the Bragg–Brentano displacement and
  Caglioti equations: https://journals.iucr.org/j/issues/1999/01/00/gl0561/
- Ida and Kimura, “Effect of sample transparency in powder diffractometry with
  Bragg–Brentano geometry as a convolution,” *Journal of Applied
  Crystallography* 32 (1999): https://doi.org/10.1107/S0021889899008894
- Dinnebier et al., “X-ray powder diffraction in education. Part I. Bragg peak
  profiles,” *Journal of Applied Crystallography* 54 (2021):
  https://journals.iucr.org/j/issues/2021/06/00/gj5272/
- GSAS-II developer documentation, constant-wavelength instrument parameters
  (`Lam1`, `Lam2`, `I(L2)/I(L1)`, `U,V,W`, `X,Y,Z`, and `SH/L`):
  https://gsas-ii.readthedocs.io/_/downloads/en/latest/pdf/
- Mendenhall et al., “High-precision measurement of the X-ray Cu K-alpha
  spectrum,” NIST (2017):
  https://www.nist.gov/publications/high-precision-measurement-x-ray-cu-k-alpha-spectrum
