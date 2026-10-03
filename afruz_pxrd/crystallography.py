from __future__ import annotations

import math
import re
import shlex
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from .instrument_calibration import (
    CalibrationRangeError,
    position_correction_deg,
    position_correction_uncertainty_deg,
)
from .instrument_physics import validate_profile_compatibility
from .gpu_backend import compute_backend


CRYSTAL_SYSTEMS = (
    "Cubic",
    "Tetragonal",
    "Orthorhombic",
    "Hexagonal",
    "Rhombohedral",
    "Monoclinic",
    "Triclinic",
)

_ELEMENT_SYMBOLS = """
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La
Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po
At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg
Cn Nh Fl Mc Lv Ts Og
""".split()
ATOMIC_NUMBER = {
    symbol: index + 1 for index, symbol in enumerate(_ELEMENT_SYMBOLS)
}


_CROMER_MANN_HEADERS = tuple(
    (f"_atom_type_scat_cromer_mann_a{index}", f"_atom_type_scat_cromer_mann_b{index}")
    for index in range(1, 5)
)


class CIFImportError(RuntimeError):
    pass


def _tokenize_cif(text: str) -> list[str]:
    tokens: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(";"):
            block = [line[1:]]
            index += 1
            while index < len(lines) and not lines[index].startswith(";"):
                block.append(lines[index])
                index += 1
            tokens.append("\n".join(block))
            index += 1
            continue

        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            try:
                # Some public CIFs use backslash-escaped quotes inside quoted
                # bibliography values (for example COD 4000663). POSIX shlex
                # treats a backslash as literal inside single quotes, so protect
                # only these two sequences while token boundaries are parsed.
                escaped_single = "\ue000"
                escaped_double = "\ue001"
                protected = stripped.replace("\\'", escaped_single).replace(
                    '\\"', escaped_double
                )
                parsed = shlex.split(protected, comments=True, posix=True)
                tokens.extend(
                    token.replace(escaped_single, "'").replace(escaped_double, '"')
                    for token in parsed
                )
            except ValueError as exc:
                raise CIFImportError(
                    f"Could not tokenize CIF line {index + 1}: {exc}"
                ) from exc
        index += 1
    return tokens


def _parse_cif_tokens(tokens: list[str]) -> tuple[dict[str, str], list[dict]]:
    tags: dict[str, str] = {}
    loops: list[dict] = []
    index = 0
    control_words = {"loop_", "stop_", "global_"}

    while index < len(tokens):
        token = tokens[index]
        lower = token.lower()

        if lower.startswith("data_") or lower.startswith("save_"):
            index += 1
            continue

        if lower == "loop_":
            index += 1
            headers = []
            while index < len(tokens) and tokens[index].startswith("_"):
                headers.append(tokens[index].lower())
                index += 1
            if not headers:
                continue

            values = []
            while index < len(tokens):
                candidate = tokens[index]
                candidate_lower = candidate.lower()
                at_row_boundary = len(values) % len(headers) == 0
                if at_row_boundary and (
                    candidate.startswith("_")
                    or candidate_lower in control_words
                    or candidate_lower.startswith("data_")
                    or candidate_lower.startswith("save_")
                ):
                    break
                values.append(candidate)
                index += 1

            rows = []
            complete_value_count = (
                len(values) // len(headers)
            ) * len(headers)
            for start in range(0, complete_value_count, len(headers)):
                rows.append(
                    dict(zip(headers, values[start:start + len(headers)]))
                )
            loops.append({"headers": headers, "rows": rows})
            continue

        if token.startswith("_"):
            value = tokens[index + 1] if index + 1 < len(tokens) else ""
            tags[token.lower()] = value
            index += 2
            continue

        index += 1

    return tags, loops


def _numeric(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text in {"", ".", "?"}:
        return default
    match = re.match(
        r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)(?:\(\d+\))?$",
        text,
    )
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def _element_from_value(value: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", value or "")
    if not letters:
        return "X"
    for length in (2, 1):
        candidate = letters[:length].capitalize()
        if candidate in ATOMIC_NUMBER:
            return candidate
    return letters[:1].capitalize()


def _find_loop(loops: list[dict], required_headers: set[str]) -> dict | None:
    for loop in loops:
        headers = set(loop["headers"])
        if required_headers.issubset(headers):
            return loop
    return None


def _parse_cif_xray_scattering_factors(loops: list[dict]) -> dict[str, dict]:
    """Read CIF-supplied Cromer--Mann X-ray scattering coefficients.

    Many experimentally refined CIFs carry the same International Tables
    coefficients used by crystallographic programs.  Retaining them avoids the
    large intensity errors caused by treating an atom as a featureless atomic
    number, while still allowing older/minimal CIFs to use the documented
    approximation.
    """

    loop = _find_loop(loops, {"_atom_type_symbol"})
    if loop is None:
        return {}
    factors: dict[str, dict] = {}
    for row in loop["rows"]:
        element = _element_from_value(row.get("_atom_type_symbol", ""))
        pairs = [
            (_numeric(row.get(a_header)), _numeric(row.get(b_header)))
            for a_header, b_header in _CROMER_MANN_HEADERS
        ]
        constant = _numeric(row.get("_atom_type_scat_cromer_mann_c"))
        if constant is None or any(a is None or b is None for a, b in pairs):
            continue
        factors[element] = {
            "a": [float(a) for a, _ in pairs],
            "b": [float(b) for _, b in pairs],
            "c": float(constant),
            "dispersion_real": float(
                _numeric(row.get("_atom_type_scat_dispersion_real"), 0.0) or 0.0
            ),
            "dispersion_imag": float(
                _numeric(row.get("_atom_type_scat_dispersion_imag"), 0.0) or 0.0
            ),
            "source": str(row.get("_atom_type_scat_source") or "CIF"),
        }
    return factors


def _normalize_crystal_system(value: str | None, cell: dict) -> str:
    text = (value or "").strip().lower()
    aliases = {
        "cubic": "Cubic",
        "tetragonal": "Tetragonal",
        "orthorhombic": "Orthorhombic",
        "hexagonal": "Hexagonal",
        "trigonal": "Rhombohedral",
        "rhombohedral": "Rhombohedral",
        "monoclinic": "Monoclinic",
        "triclinic": "Triclinic",
    }
    if text in aliases:
        return aliases[text]

    a, b, c = cell["a"], cell["b"], cell["c"]
    alpha, beta, gamma = cell["alpha"], cell["beta"], cell["gamma"]
    close = lambda x, y, tolerance=1e-3: abs(x - y) <= tolerance
    angle_close = lambda x, y, tolerance=0.05: abs(x - y) <= tolerance

    if close(a, b) and close(b, c):
        if all(angle_close(angle, 90.0) for angle in (alpha, beta, gamma)):
            return "Cubic"
        if close(alpha, beta) and close(beta, gamma):
            return "Rhombohedral"
    if close(a, b) and all(
        angle_close(angle, 90.0) for angle in (alpha, beta)
    ):
        if angle_close(gamma, 120.0):
            return "Hexagonal"
        if angle_close(gamma, 90.0):
            return "Tetragonal"
    if all(angle_close(angle, 90.0) for angle in (alpha, beta, gamma)):
        return "Orthorhombic"
    if angle_close(alpha, 90.0) and angle_close(gamma, 90.0):
        return "Monoclinic"
    return "Triclinic"


def _safe_symmetry_expression(expression: str, x: float, y: float, z: float) -> float:
    expression = expression.strip().lower()
    if not re.fullmatch(r"[xyz0-9+\-*/().\s]+", expression):
        raise CIFImportError(
            f"Unsupported or unsafe symmetry expression: {expression!r}"
        )
    try:
        return float(
            eval(
                expression,
                {"__builtins__": {}},
                {"x": x, "y": y, "z": z},
            )
        )
    except Exception as exc:
        raise CIFImportError(
            f"Could not evaluate symmetry expression {expression!r}: {exc}"
        ) from exc


def _apply_symmetry_operation(
    operation: str,
    coordinates: tuple[float, float, float],
) -> tuple[float, float, float]:
    parts = [part.strip() for part in operation.split(",")]
    if len(parts) != 3:
        raise CIFImportError(
            f"Symmetry operation must contain three coordinates: {operation!r}"
        )
    x, y, z = coordinates
    result = tuple(
        _safe_symmetry_expression(part, x, y, z) % 1.0
        for part in parts
    )
    return result


def _expand_atoms(
    asymmetric_atoms: list[dict],
    symmetry_operations: list[str],
) -> list[dict]:
    operations = symmetry_operations or ["x,y,z"]
    expanded: list[dict] = []
    seen = set()

    for atom in asymmetric_atoms:
        coordinates = (atom["x"], atom["y"], atom["z"])
        for operation in operations:
            transformed = _apply_symmetry_operation(operation, coordinates)
            # Published high-symmetry CIFs commonly round special-position
            # coordinates to 4–5 decimal places. Equivalent operations can
            # therefore differ by ~1e-5 in fractional coordinates even though
            # they represent the same site. Deduplicate within each labelled
            # asymmetric site at 1e-3 fractional precision; different CIF site
            # labels remain distinct so disorder models are preserved.
            key = (
                atom.get("label", atom["element"]),
                round(transformed[0] % 1.0, 3),
                round(transformed[1] % 1.0, 3),
                round(transformed[2] % 1.0, 3),
            )
            if key in seen:
                continue
            seen.add(key)
            copy = dict(atom)
            copy.update(
                {
                    "x": float(transformed[0] % 1.0),
                    "y": float(transformed[1] % 1.0),
                    "z": float(transformed[2] % 1.0),
                }
            )
            expanded.append(copy)
    return expanded


def parse_cif_text(text: str, source_path: str = "") -> dict:
    tokens = _tokenize_cif(text)
    tags, loops = _parse_cif_tokens(tokens)
    xray_scattering_factors = _parse_cif_xray_scattering_factors(loops)

    cell = {
        "a": _numeric(tags.get("_cell_length_a")),
        "b": _numeric(tags.get("_cell_length_b")),
        "c": _numeric(tags.get("_cell_length_c")),
        "alpha": _numeric(tags.get("_cell_angle_alpha")),
        "beta": _numeric(tags.get("_cell_angle_beta")),
        "gamma": _numeric(tags.get("_cell_angle_gamma")),
    }
    missing = [name for name, value in cell.items() if value is None]
    if missing:
        raise CIFImportError(
            "CIF is missing required unit-cell values: " + ", ".join(missing)
        )
    cell = {name: float(value) for name, value in cell.items()}

    atom_loop = _find_loop(
        loops,
        {
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        },
    )
    asymmetric_atoms = []
    if atom_loop is not None:
        for row_index, row in enumerate(atom_loop["rows"], start=1):
            x = _numeric(row.get("_atom_site_fract_x"))
            y = _numeric(row.get("_atom_site_fract_y"))
            z = _numeric(row.get("_atom_site_fract_z"))
            if None in (x, y, z):
                continue
            symbol_source = (
                row.get("_atom_site_type_symbol")
                or row.get("_atom_site_label")
                or "X"
            )
            element = _element_from_value(symbol_source)
            occupancy = _numeric(row.get("_atom_site_occupancy"), 1.0)
            if occupancy is not None:
                # Accept harmless CIF rounding such as 1.00002 while retaining
                # genuinely invalid or intentionally unusual values for the
                # downstream structure validator to flag.
                if 1.0 < occupancy <= 1.001:
                    occupancy = 1.0
                elif -0.001 <= occupancy < 0.0:
                    occupancy = 0.0
            b_iso = _numeric(row.get("_atom_site_b_iso_or_equiv"))
            u_iso = _numeric(row.get("_atom_site_u_iso_or_equiv"))
            charge = _numeric(row.get("_atom_site_charge"))
            if b_iso is None and u_iso is not None:
                b_iso = 8.0 * math.pi * math.pi * u_iso
            atom = {
                "label": row.get(
                    "_atom_site_label",
                    f"{element}{row_index}",
                ),
                "element": element,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "occupancy": float(1.0 if occupancy is None else occupancy),
                "b_iso": float(0.0 if b_iso is None else b_iso),
            }
            if charge is not None:
                # Preserve CIF/DDEC partial charges as metadata. They are not
                # interpreted as oxidation states or used for bond inference.
                atom["charge"] = float(charge)
            asymmetric_atoms.append(atom)

    symmetry_operations = []
    for possible_header in (
        "_space_group_symop_operation_xyz",
        "_symmetry_equiv_pos_as_xyz",
    ):
        loop = _find_loop(loops, {possible_header})
        if loop is not None:
            symmetry_operations = [
                row[possible_header] for row in loop["rows"]
                if row.get(possible_header)
            ]
            break
    if not symmetry_operations:
        single_operation = (
            tags.get("_space_group_symop_operation_xyz")
            or tags.get("_symmetry_equiv_pos_as_xyz")
        )
        if single_operation:
            symmetry_operations = [single_operation]
    if not symmetry_operations:
        symmetry_operations = ["x,y,z"]

    expanded_atoms = _expand_atoms(asymmetric_atoms, symmetry_operations)

    # Preserve any author-supplied geometric bond records.  These records are
    # useful display evidence, but their symmetry codes are intentionally kept
    # verbatim: resolving arbitrary CIF symmetry references is a separate
    # crystallographic operation and must not be guessed by the renderer.
    cif_bonds = []
    bond_loop = _find_loop(
        loops,
        {
            "_geom_bond_atom_site_label_1",
            "_geom_bond_atom_site_label_2",
        },
    )
    if bond_loop is not None:
        for row in bond_loop["rows"]:
            first = str(row.get("_geom_bond_atom_site_label_1") or "").strip()
            second = str(row.get("_geom_bond_atom_site_label_2") or "").strip()
            if not first or not second or first in {".", "?"} or second in {".", "?"}:
                continue
            record = {
                "label_1": first,
                "label_2": second,
                "distance": _numeric(row.get("_geom_bond_distance")),
                "symmetry_1": str(
                    row.get("_geom_bond_site_symmetry_1") or "."
                ),
                "symmetry_2": str(
                    row.get("_geom_bond_site_symmetry_2") or "."
                ),
                "type": str(row.get("_geom_bond_type") or ""),
            }
            cif_bonds.append(record)

    crystal_system_tag = (
        tags.get("_space_group_crystal_system")
        or tags.get("_symmetry_cell_setting")
    )
    crystal_system = _normalize_crystal_system(crystal_system_tag, cell)
    space_group = (
        tags.get("_space_group_name_h-m_alt")
        or tags.get("_symmetry_space_group_name_h-m")
        or tags.get("_space_group_name_hall")
        or "Unknown"
    )

    return {
        "source_path": source_path,
        "raw_cif_text": text,
        "data_name": next(
            (
                token[5:]
                for token in tokens
                if token.lower().startswith("data_")
            ),
            Path(source_path).stem if source_path else "CIF structure",
        ),
        "formula": (
            tags.get("_chemical_formula_sum")
            or tags.get("_chemical_formula_structural")
            or ""
        ),
        "formula_units_z": _numeric(tags.get("_cell_formula_units_z")),
        "formula_weight": _numeric(tags.get("_chemical_formula_weight")),
        "space_group": space_group,
        "crystal_system": crystal_system,
        "cell": cell,
        "symmetry_operations": symmetry_operations,
        "asymmetric_atom_count": len(asymmetric_atoms),
        "expanded_atom_count": len(expanded_atoms),
        "atoms": expanded_atoms,
        "cif_bonds": cif_bonds,
        "xray_scattering_factors": xray_scattering_factors,
        "reported_wavelength_angstrom": _numeric(
            tags.get("_pd_proc_wavelength")
            or tags.get("_diffrn_radiation_wavelength")
        ),
        "intensity_model": (
            "CIF-supplied Cromer–Mann X-ray form factors, anomalous dispersion, "
            "isotropic Debye–Waller attenuation, multiplicity, and "
            "Lorentz-polarization weighting."
            if xray_scattering_factors
            else "Approximate kinematic structure factors using atomic number, "
            "isotropic Debye–Waller attenuation, multiplicity, and "
            "Lorentz-polarization weighting (the CIF supplies no Cromer–Mann table)."
        ),
    }


def load_cif(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise CIFImportError(f"CIF file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CIFImportError(f"Could not read CIF file: {exc}") from exc
    return parse_cif_text(text, source_path=str(path.resolve()))


def ensure_xray_scattering_metadata(structure: dict) -> dict[str, dict]:
    """Hydrate scattering metadata in projects saved by older releases."""

    existing = structure.get("xray_scattering_factors")
    if isinstance(existing, dict) and existing:
        return existing
    raw_text = str(structure.get("raw_cif_text") or "")
    if not raw_text:
        structure.setdefault("xray_scattering_factors", {})
        return {}
    try:
        tags, loops = _parse_cif_tokens(_tokenize_cif(raw_text))
        factors = _parse_cif_xray_scattering_factors(loops)
    except CIFImportError:
        factors = {}
        tags = {}
    structure["xray_scattering_factors"] = factors
    if structure.get("reported_wavelength_angstrom") is None:
        structure["reported_wavelength_angstrom"] = _numeric(
            tags.get("_pd_proc_wavelength")
            or tags.get("_diffrn_radiation_wavelength")
        )
    if factors:
        structure["intensity_model"] = (
            "CIF-supplied Cromer–Mann X-ray form factors, anomalous dispersion, "
            "isotropic Debye–Waller attenuation, multiplicity, and "
            "Lorentz-polarization weighting."
        )
    return factors


def direct_metric_tensor(cell: dict) -> np.ndarray:
    a, b, c = float(cell["a"]), float(cell["b"]), float(cell["c"])
    alpha = math.radians(float(cell["alpha"]))
    beta = math.radians(float(cell["beta"]))
    gamma = math.radians(float(cell["gamma"]))

    metric = np.asarray(
        [
            [a * a, a * b * math.cos(gamma), a * c * math.cos(beta)],
            [a * b * math.cos(gamma), b * b, b * c * math.cos(alpha)],
            [a * c * math.cos(beta), b * c * math.cos(alpha), c * c],
        ],
        dtype=float,
    )
    determinant = float(np.linalg.det(metric))
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("Unit-cell metric is not physically valid.")
    return metric


def d_spacing(cell: dict, hkl: tuple[int, int, int]) -> float:
    hkl_vector = np.asarray(hkl, dtype=float)
    reciprocal_metric = np.linalg.inv(direct_metric_tensor(cell))
    inverse_d_squared = float(
        hkl_vector @ reciprocal_metric @ hkl_vector
    )
    if inverse_d_squared <= 0:
        raise ValueError("Reflection has a non-positive reciprocal spacing.")
    return float(1.0 / math.sqrt(inverse_d_squared))


def two_theta_from_hkl(
    cell: dict,
    hkl: tuple[int, int, int],
    wavelength_angstrom: float,
) -> float | None:
    d_value = d_spacing(cell, hkl)
    sine_theta = float(wavelength_angstrom) / (2.0 * d_value)
    if not 0.0 < sine_theta < 1.0:
        return None
    return float(2.0 * math.degrees(math.asin(sine_theta)))


def _canonical_hkl(hkl: tuple[int, int, int]) -> tuple[int, int, int]:
    values = list(hkl)
    for value in values:
        if value < 0:
            return tuple(-component for component in values)
        if value > 0:
            return tuple(values)
    return tuple(values)


def _xray_scattering_factor(
    element: str,
    sine_theta_over_lambda_squared: float,
    scattering_factors: dict[str, dict] | None = None,
) -> complex:
    """Return f0 + f' + i f'' for one element.

    The four-Gaussian Cromer--Mann expression is used when the CIF supplies
    coefficients.  The legacy atomic-number approximation remains an explicit
    fallback for minimal CIFs without an atom-type scattering table.
    """

    s2 = max(0.0, float(sine_theta_over_lambda_squared))
    parameters = (scattering_factors or {}).get(str(element))
    if parameters:
        a_values = parameters.get("a") or ()
        b_values = parameters.get("b") or ()
        if len(a_values) == 4 and len(b_values) == 4:
            f0 = float(parameters.get("c", 0.0)) + sum(
                float(a) * math.exp(-float(b) * s2)
                for a, b in zip(a_values, b_values)
            )
            return complex(
                f0 + float(parameters.get("dispersion_real", 0.0)),
                float(parameters.get("dispersion_imag", 0.0)),
            )
    atomic_number = float(ATOMIC_NUMBER.get(str(element), 1))
    return complex(atomic_number * math.exp(-2.0 * s2), 0.0)


def _structure_factor_intensity(
    atoms: list[dict],
    hkl: tuple[int, int, int],
    d_value: float,
    two_theta_deg: float,
    wavelength_angstrom: float,
    scattering_factors: dict[str, dict] | None = None,
) -> float:
    if not atoms:
        return 1.0

    h, k, l = hkl
    sine_theta_over_lambda = 1.0 / (2.0 * d_value)
    s2 = sine_theta_over_lambda ** 2
    structure_factor = 0.0 + 0.0j
    for atom in atoms:
        xray_scattering = _xray_scattering_factor(
            atom["element"],
            s2,
            scattering_factors,
        )
        debye_waller = math.exp(
            -max(0.0, float(atom.get("b_iso", 0.0)))
            * s2
        )
        phase = 2.0 * math.pi * (
            h * float(atom["x"])
            + k * float(atom["y"])
            + l * float(atom["z"])
        )
        structure_factor += (
            float(atom.get("occupancy", 1.0))
            * xray_scattering
            * debye_waller
            * complex(math.cos(phase), math.sin(phase))
        )

    intensity = abs(structure_factor) ** 2
    theta = math.radians(two_theta_deg / 2.0)
    two_theta = math.radians(two_theta_deg)
    denominator = max(
        math.sin(theta) ** 2 * abs(math.cos(theta)),
        1e-8,
    )
    lorentz_polarization = (
        1.0 + math.cos(two_theta) ** 2
    ) / denominator
    return float(intensity * min(lorentz_polarization, 1e6))


def _structure_factor_intensities_batch(
    atoms: list[dict],
    hkls: Iterable[tuple[int, int, int]],
    d_values: Iterable[float],
    two_theta_values: Iterable[float],
    *,
    use_gpu: bool = False,
    scattering_factors: dict[str, dict] | None = None,
) -> tuple[np.ndarray, dict]:
    """Evaluate independent structure factors in CPU/GPU matrix batches.

    The formula is deliberately identical to :func:`_structure_factor_intensity`.
    Only the execution backend changes, so enabling CUDA cannot change which
    physical terms are included in the calculation.
    """
    hkl_array = np.asarray(list(hkls), dtype=float).reshape(-1, 3)
    d_array = np.asarray(list(d_values), dtype=float)
    tt_array = np.asarray(list(two_theta_values), dtype=float)
    if not (len(hkl_array) == len(d_array) == len(tt_array)):
        raise ValueError("hkl, d-spacing and two-theta batches must have equal lengths.")
    if not len(hkl_array):
        return np.asarray([], dtype=float), {
            "requested": bool(use_gpu), "used": False, "backend": "CPU",
            "device_name": "", "reason": "The reflection batch is empty.",
        }
    if not atoms:
        return np.ones(len(hkl_array), dtype=float), {
            "requested": bool(use_gpu), "used": False, "backend": "CPU",
            "device_name": "", "reason": "The atomic model is empty.",
        }

    coordinates = np.asarray(
        [[atom[axis] for axis in ("x", "y", "z")] for atom in atoms],
        dtype=float,
    )
    occupancies = np.asarray(
        [float(atom.get("occupancy", 1.0)) for atom in atoms], dtype=float,
    )
    atomic_numbers = np.asarray(
        [float(ATOMIC_NUMBER.get(atom.get("element", "X"), 1)) for atom in atoms],
        dtype=float,
    )
    cromer_a = np.zeros((len(atoms), 4), dtype=float)
    cromer_b = np.zeros((len(atoms), 4), dtype=float)
    cromer_c = np.zeros(len(atoms), dtype=float)
    dispersion_real = np.zeros(len(atoms), dtype=float)
    dispersion_imag = np.zeros(len(atoms), dtype=float)
    has_cromer_mann = np.zeros(len(atoms), dtype=bool)
    for atom_index, atom in enumerate(atoms):
        parameters = (scattering_factors or {}).get(str(atom.get("element", "X")))
        if not parameters:
            continue
        a_values = parameters.get("a") or ()
        b_values = parameters.get("b") or ()
        if len(a_values) != 4 or len(b_values) != 4:
            continue
        cromer_a[atom_index, :] = np.asarray(a_values, dtype=float)
        cromer_b[atom_index, :] = np.asarray(b_values, dtype=float)
        cromer_c[atom_index] = float(parameters.get("c", 0.0))
        dispersion_real[atom_index] = float(
            parameters.get("dispersion_real", 0.0)
        )
        dispersion_imag[atom_index] = float(
            parameters.get("dispersion_imag", 0.0)
        )
        has_cromer_mann[atom_index] = True
    b_iso = np.maximum(
        np.asarray([float(atom.get("b_iso", 0.0)) for atom in atoms], dtype=float),
        0.0,
    )

    def evaluate(xp, status):
        coords = xp.asarray(coordinates, dtype=xp.float64)
        occupancy = xp.asarray(occupancies, dtype=xp.float64)
        atomic_z = xp.asarray(atomic_numbers, dtype=xp.float64)
        cm_a = xp.asarray(cromer_a, dtype=xp.float64)
        cm_b = xp.asarray(cromer_b, dtype=xp.float64)
        cm_c = xp.asarray(cromer_c, dtype=xp.float64)
        fp = xp.asarray(dispersion_real, dtype=xp.float64)
        fpp = xp.asarray(dispersion_imag, dtype=xp.float64)
        cm_mask = xp.asarray(has_cromer_mann, dtype=xp.bool_)
        thermal = xp.asarray(b_iso, dtype=xp.float64)
        result = np.empty(len(hkl_array), dtype=float)
        # This keeps peak memory modest for 10k–20k atom expanded MOFs while
        # still giving CUDA enough parallel work per kernel launch.
        chunk_size = 128 if xp is not np else 48
        for start in range(0, len(hkl_array), chunk_size):
            stop = min(len(hkl_array), start + chunk_size)
            local_hkl = xp.asarray(hkl_array[start:stop], dtype=xp.float64)
            s2 = xp.asarray(1.0 / (4.0 * d_array[start:stop] ** 2), dtype=xp.float64)
            fallback_f0 = atomic_z[:, None] * xp.exp(-2.0 * s2[None, :])
            tabulated_f0 = cm_c[:, None]
            for term in range(4):
                tabulated_f0 = tabulated_f0 + cm_a[:, term, None] * xp.exp(
                    -cm_b[:, term, None] * s2[None, :]
                )
            f_real = xp.where(cm_mask[:, None], tabulated_f0 + fp[:, None], fallback_f0)
            f_imag = xp.where(cm_mask[:, None], fpp[:, None], 0.0)
            attenuation = occupancy[:, None] * xp.exp(-thermal[:, None] * s2[None, :])
            f_real = f_real * attenuation
            f_imag = f_imag * attenuation
            phase = 2.0 * math.pi * (coords @ local_hkl.T)
            cos_phase = xp.cos(phase)
            sin_phase = xp.sin(phase)
            real = xp.sum(f_real * cos_phase - f_imag * sin_phase, axis=0)
            imaginary = xp.sum(f_real * sin_phase + f_imag * cos_phase, axis=0)
            theta = xp.asarray(np.radians(tt_array[start:stop] / 2.0), dtype=xp.float64)
            two_theta = xp.asarray(np.radians(tt_array[start:stop]), dtype=xp.float64)
            denominator = xp.maximum(
                xp.sin(theta) ** 2 * xp.abs(xp.cos(theta)), 1e-8
            )
            lorentz_polarization = xp.minimum(
                (1.0 + xp.cos(two_theta) ** 2) / denominator, 1e6
            )
            values = (real * real + imaginary * imaginary) * lorentz_polarization
            if xp is np:
                result[start:stop] = np.asarray(values, dtype=float)
            else:
                result[start:stop] = xp.asnumpy(values)
        status = dict(status)
        status.update(requested=bool(use_gpu), used=xp is not np)
        return result, status

    xp, status = compute_backend(use_gpu)
    try:
        return evaluate(xp, status)
    except Exception as exc:
        if xp is np:
            raise
        fallback = dict(status)
        fallback.update(
            available=False,
            backend="CPU",
            reason=f"CUDA calculation failed; CPU fallback used: {exc}",
        )
        return evaluate(np, fallback)


def _large_fd3m_powder_pattern(
    structure: dict,
    wavelength_angstrom: float,
    two_theta_min: float,
    two_theta_max: float,
    intensity_cutoff_percent: float,
    merge_tolerance_deg: float,
    use_gpu: bool = False,
) -> list[dict]:
    """Vectorized powder pattern for very large cubic Fd-3m models.

    Exhaustively visiting every signed hkl and then looping over >10,000
    expanded atoms is prohibitively slow.  For cubic Fd-3m, enumerate one
    representative of each permutation/sign orbit, apply the F/diamond
    reflection conditions, and restore the exact cubic multiplicity.  The
    structure factors remain calculated from every expanded CIF atom.
    """
    cell = structure["cell"]
    atoms = structure.get("atoms", [])
    d_min = wavelength_angstrom / (
        2.0 * math.sin(math.radians(two_theta_max / 2.0))
    )
    # Do not truncate large-cell MOFs at h,k,l=40.  A ~89 A MIL-101 cell
    # requires indices close to 100 to cover a Cu K-alpha pattern to 120 deg.
    # The symmetry-reduced Fd-3m enumeration keeps that complete range tractable.
    maximum_index = max(1, int(math.ceil(float(cell["a"]) / d_min)) + 2)

    representatives: list[tuple[int, int, int]] = []
    multiplicities: list[int] = []
    d_values: list[float] = []
    two_theta_values: list[float] = []
    for h in range(maximum_index + 1):
        for k in range(h + 1):
            for l in range(k + 1):
                if h == k == l == 0:
                    continue
                parity = (h % 2, k % 2, l % 2)
                if not (parity == (1, 1, 1) or parity == (0, 0, 0)):
                    continue
                if parity == (0, 0, 0) and (h + k + l) % 4:
                    continue
                hkl = (h, k, l)
                try:
                    d_value = d_spacing(cell, hkl)
                except ValueError:
                    continue
                sine_theta = wavelength_angstrom / (2.0 * d_value)
                if not 0.0 < sine_theta < 1.0:
                    continue
                two_theta = 2.0 * math.degrees(math.asin(sine_theta))
                if not two_theta_min <= two_theta <= two_theta_max:
                    continue
                counts = [hkl.count(value) for value in set(hkl)]
                permutation_count = math.factorial(3)
                for count in counts:
                    permutation_count //= math.factorial(count)
                sign_count = 2 ** sum(value != 0 for value in hkl)
                representatives.append(hkl)
                multiplicities.append(permutation_count * sign_count)
                d_values.append(float(d_value))
                two_theta_values.append(float(two_theta))

    if not representatives:
        return []
    multiplicity_array = np.asarray(multiplicities, dtype=float)
    raw_intensities, acceleration = _structure_factor_intensities_batch(
        atoms,
        representatives,
        d_values,
        two_theta_values,
        use_gpu=use_gpu,
        scattering_factors=structure.get("xray_scattering_factors"),
    )
    raw_intensities *= multiplicity_array

    groups: dict[int, dict] = {}
    for hkl, multiplicity, two_theta, intensity in zip(
        representatives,
        multiplicities,
        two_theta_values,
        raw_intensities,
    ):
        if not np.isfinite(intensity) or intensity <= 1e-12:
            continue
        # Cubic reflections are exactly coincident when h^2+k^2+l^2 is equal.
        # Group by that crystallographic invariant rather than an arbitrary
        # angular bin, which can merge distinct high-angle MIL-101 reflections.
        group_key = sum(component * component for component in hkl)
        group = groups.setdefault(
            group_key,
            {
                "weighted_two_theta": 0.0,
                "intensity": 0.0,
                "hkl_intensities": {},
                "hkl_multiplicities": {},
            },
        )
        group["weighted_two_theta"] += two_theta * float(intensity)
        group["intensity"] += float(intensity)
        group["hkl_intensities"][hkl] = float(intensity)
        group["hkl_multiplicities"][hkl] = int(multiplicity)

    peaks = []
    for group in groups.values():
        total_intensity = float(group["intensity"])
        if total_intensity <= 0.0:
            continue
        two_theta = float(group["weighted_two_theta"] / total_intensity)
        theta = math.radians(two_theta / 2.0)
        d_value = wavelength_angstrom / (2.0 * math.sin(theta))
        ordered_hkls = sorted(
            group["hkl_intensities"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        primary_hkl = ordered_hkls[0][0]
        retained = ordered_hkls
        peak = {
                "two_theta": two_theta,
                "d_spacing": float(d_value),
                "intensity_raw": total_intensity,
                "intensity": total_intensity,
                "hkl": list(primary_hkl),
                "hkl_label": f"({primary_hkl[0]} {primary_hkl[1]} {primary_hkl[2]})",
                "multiplicity_count": int(
                    sum(group["hkl_multiplicities"].values())
                ),
                "equivalent_hkls": [list(hkl) for hkl, _ in retained],
                "equivalent_multiplicities": [
                    int(group["hkl_multiplicities"][hkl])
                    for hkl, _ in retained
                ],
                "large_structure_fast_path": True,
            }
        if use_gpu:
            peak["acceleration_backend"] = acceleration.get("backend", "CPU")
            peak["acceleration_device"] = acceleration.get("device_name", "")
        peaks.append(peak)
    if not peaks:
        return []
    maximum = max(peak["intensity"] for peak in peaks)
    for peak in peaks:
        peak["intensity"] = float(100.0 * peak["intensity"] / maximum)
    cutoff = max(0.0, float(intensity_cutoff_percent))
    peaks = [peak for peak in peaks if peak["intensity"] >= cutoff]
    peaks.sort(key=lambda peak: peak["two_theta"])
    return peaks


def calculate_powder_pattern(
    structure: dict,
    wavelength_angstrom: float,
    two_theta_min: float,
    two_theta_max: float,
    intensity_cutoff_percent: float = 0.5,
    merge_tolerance_deg: float = 0.015,
    use_gpu: bool = False,
) -> list[dict]:
    wavelength_angstrom = float(wavelength_angstrom)
    two_theta_min = float(two_theta_min)
    two_theta_max = float(two_theta_max)
    if wavelength_angstrom <= 0:
        raise ValueError("Wavelength must be positive.")
    if not 0 <= two_theta_min < two_theta_max < 180:
        raise ValueError("2θ range must satisfy 0 ≤ minimum < maximum < 180°.")
    ensure_xray_scattering_metadata(structure)

    normalized_space_group = re.sub(
        r"[\s:_-]+",
        "",
        str(structure.get("space_group", "")),
    ).lower()
    if (
        str(structure.get("crystal_system", "")) == "Cubic"
        and normalized_space_group.startswith("fd3m")
        and len(structure.get("atoms", [])) > 2000
    ):
        return _large_fd3m_powder_pattern(
            structure,
            wavelength_angstrom,
            two_theta_min,
            two_theta_max,
            intensity_cutoff_percent,
            merge_tolerance_deg,
            use_gpu=use_gpu,
        )

    cell = structure["cell"]
    d_min = wavelength_angstrom / (
        2.0 * math.sin(math.radians(two_theta_max / 2.0))
    )
    index_limits = [
        min(40, max(1, int(math.ceil(float(cell[name]) / d_min)) + 2))
        for name in ("a", "b", "c")
    ]

    groups: dict[int, dict] = {}
    atoms = structure.get("atoms", [])

    for h in range(-index_limits[0], index_limits[0] + 1):
        for k in range(-index_limits[1], index_limits[1] + 1):
            for l in range(-index_limits[2], index_limits[2] + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl = (h, k, l)
                try:
                    d_value = d_spacing(cell, hkl)
                except ValueError:
                    continue
                sine_theta = wavelength_angstrom / (2.0 * d_value)
                if not 0.0 < sine_theta < 1.0:
                    continue
                two_theta = 2.0 * math.degrees(math.asin(sine_theta))
                if not two_theta_min <= two_theta <= two_theta_max:
                    continue

                individual_intensity = _structure_factor_intensity(
                    atoms,
                    hkl,
                    d_value,
                    two_theta,
                    wavelength_angstrom,
                    structure.get("xray_scattering_factors"),
                )
                if individual_intensity <= 1e-12:
                    continue

                group_key = int(round(two_theta / merge_tolerance_deg))
                group = groups.setdefault(
                    group_key,
                    {
                        "weighted_two_theta": 0.0,
                        "intensity": 0.0,
                        "hkl_intensities": {},
                        "hkl_multiplicities": {},
                    },
                )
                group["weighted_two_theta"] += (
                    two_theta * individual_intensity
                )
                group["intensity"] += individual_intensity
                canonical = _canonical_hkl(hkl)
                group["hkl_intensities"][canonical] = (
                    group["hkl_intensities"].get(canonical, 0.0)
                    + individual_intensity
                )
                group["hkl_multiplicities"][canonical] = (
                    group["hkl_multiplicities"].get(canonical, 0) + 1
                )

    peaks = []
    for group in groups.values():
        total_intensity = float(group["intensity"])
        if total_intensity <= 0:
            continue
        two_theta = float(
            group["weighted_two_theta"] / total_intensity
        )
        theta = math.radians(two_theta / 2.0)
        d_value = wavelength_angstrom / (2.0 * math.sin(theta))
        ordered_hkls = sorted(
            group["hkl_intensities"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
        primary_hkl = ordered_hkls[0][0]
        peaks.append(
            {
                "two_theta": two_theta,
                "d_spacing": float(d_value),
                "intensity_raw": total_intensity,
                "intensity": total_intensity,
                "hkl": list(primary_hkl),
                "hkl_label": (
                    f"({primary_hkl[0]} {primary_hkl[1]} {primary_hkl[2]})"
                ),
                "multiplicity_count": int(
                    sum(group["hkl_multiplicities"].values())
                ),
                "equivalent_hkls": [
                    list(hkl) for hkl, _ in ordered_hkls
                ],
                "equivalent_multiplicities": [
                    int(group["hkl_multiplicities"][hkl])
                    for hkl, _ in ordered_hkls
                ],
            }
        )

    if not peaks:
        return []

    maximum = max(peak["intensity"] for peak in peaks)
    for peak in peaks:
        peak["intensity"] = float(
            100.0 * peak["intensity"] / maximum
        )

    cutoff = max(0.0, float(intensity_cutoff_percent))
    peaks = [
        peak for peak in peaks if peak["intensity"] >= cutoff
    ]
    peaks.sort(key=lambda peak: peak["two_theta"])
    return peaks


def match_observed_to_reference(
    observed_peaks: list[dict],
    reference_peaks: list[dict],
    tolerance_deg: float,
) -> list[dict]:
    tolerance_deg = max(0.0, float(tolerance_deg))
    candidates = []
    for observed_index, observed in enumerate(observed_peaks):
        observed_position = float(observed["position"])
        for reference_index, reference in enumerate(reference_peaks):
            delta = observed_position - float(reference["two_theta"])
            if abs(delta) <= tolerance_deg:
                candidates.append(
                    (
                        abs(delta),
                        observed_index,
                        reference_index,
                        delta,
                    )
                )

    used_observed = set()
    used_reference = set()
    matches = []
    for _, observed_index, reference_index, delta in sorted(candidates):
        if observed_index in used_observed or reference_index in used_reference:
            continue
        used_observed.add(observed_index)
        used_reference.add(reference_index)

        observed = observed_peaks[observed_index]
        reference = reference_peaks[reference_index]
        matches.append(
            {
                "observed_index": observed_index,
                "reference_index": reference_index,
                "observed_2theta": float(observed["position"]),
                "observed_error": observed.get("position_error"),
                "reference_2theta": float(reference["two_theta"]),
                "initial_delta": float(delta),
                "hkl": list(reference["hkl"]),
                "hkl_label": reference["hkl_label"],
                "reference_intensity": float(reference["intensity"]),
            }
        )

    matches.sort(key=lambda row: row["observed_2theta"])
    return matches


def _cell_parameters_for_system(cell: dict, crystal_system: str):
    system = crystal_system
    if system == "Cubic":
        return ["a"], [cell["a"]]
    if system == "Tetragonal":
        return ["a", "c"], [cell["a"], cell["c"]]
    if system == "Orthorhombic":
        return ["a", "b", "c"], [cell["a"], cell["b"], cell["c"]]
    if system == "Hexagonal":
        return ["a", "c"], [cell["a"], cell["c"]]
    if system == "Rhombohedral":
        return ["a", "alpha"], [cell["a"], cell["alpha"]]
    if system == "Monoclinic":
        return ["a", "b", "c", "beta"], [
            cell["a"], cell["b"], cell["c"], cell["beta"]
        ]
    return ["a", "b", "c", "alpha", "beta", "gamma"], [
        cell["a"],
        cell["b"],
        cell["c"],
        cell["alpha"],
        cell["beta"],
        cell["gamma"],
    ]


def _cell_from_system_parameters(
    parameters: dict[str, float],
    crystal_system: str,
) -> dict:
    if crystal_system == "Cubic":
        a = parameters["a"]
        return {
            "a": a, "b": a, "c": a,
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if crystal_system == "Tetragonal":
        return {
            "a": parameters["a"], "b": parameters["a"], "c": parameters["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if crystal_system == "Orthorhombic":
        return {
            "a": parameters["a"], "b": parameters["b"], "c": parameters["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
        }
    if crystal_system == "Hexagonal":
        return {
            "a": parameters["a"], "b": parameters["a"], "c": parameters["c"],
            "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
        }
    if crystal_system == "Rhombohedral":
        return {
            "a": parameters["a"], "b": parameters["a"], "c": parameters["a"],
            "alpha": parameters["alpha"],
            "beta": parameters["alpha"],
            "gamma": parameters["alpha"],
        }
    if crystal_system == "Monoclinic":
        return {
            "a": parameters["a"], "b": parameters["b"], "c": parameters["c"],
            "alpha": 90.0, "beta": parameters["beta"], "gamma": 90.0,
        }
    return {
        name: parameters[name]
        for name in ("a", "b", "c", "alpha", "beta", "gamma")
    }


def refine_unit_cell(
    matches: list[dict],
    initial_cell: dict,
    crystal_system: str,
    wavelength_angstrom: float,
    refine_zero_shift: bool = True,
    instrument_profile: dict | None = None,
    apply_instrument_position_correction: bool = False,
    allow_profile_extrapolation: bool = False,
) -> dict:
    if crystal_system not in CRYSTAL_SYSTEMS:
        raise ValueError(f"Unsupported crystal system: {crystal_system}")
    if wavelength_angstrom <= 0:
        raise ValueError("Wavelength must be positive.")
    if instrument_profile:
        validate_profile_compatibility(
            instrument_profile,
            wavelength_angstrom=wavelength_angstrom,
        )

    parameter_names, initial_values = _cell_parameters_for_system(
        initial_cell,
        crystal_system,
    )
    if refine_zero_shift:
        parameter_names.append("zero_shift")
        initial_values.append(0.0)

    minimum_matches = len(parameter_names) + 1
    if len(matches) < minimum_matches:
        raise ValueError(
            f"At least {minimum_matches} uniquely matched peaks are required "
            f"to refine {len(parameter_names)} parameter(s)."
        )

    lower_bounds = []
    upper_bounds = []
    for name, value in zip(parameter_names, initial_values):
        if name in {"a", "b", "c"}:
            lower_bounds.append(max(0.1, 0.5 * float(value)))
            upper_bounds.append(1.5 * float(value))
        elif name in {"alpha", "beta", "gamma"}:
            lower_bounds.append(35.0)
            upper_bounds.append(145.0)
        elif name == "zero_shift":
            lower_bounds.append(-2.0)
            upper_bounds.append(2.0)

    raw_observed = np.asarray(
        [match["observed_2theta"] for match in matches],
        dtype=float,
    )
    errors = np.asarray(
        [
            np.nan
            if match.get("observed_error") is None
            else float(match["observed_error"])
            for match in matches
        ],
        dtype=float,
    )
    use_weights = bool(
        np.all(np.isfinite(errors)) and np.all(errors > 0)
    )
    calibration_corrections = np.zeros_like(raw_observed)
    calibration_errors = np.zeros_like(raw_observed)
    if apply_instrument_position_correction:
        if not instrument_profile:
            raise ValueError(
                "An instrument profile is required to correct cell-refinement positions."
            )
        try:
            calibration_corrections = np.asarray(
                position_correction_deg(raw_observed, instrument_profile),
                dtype=float,
            )
            calibration_errors = np.asarray(
                [
                    position_correction_uncertainty_deg(
                        value,
                        instrument_profile,
                        allow_extrapolation=allow_profile_extrapolation,
                    )
                    for value in raw_observed
                ],
                dtype=float,
            )
        except CalibrationRangeError as exc:
            raise ValueError(str(exc)) from exc
        observed = raw_observed - calibration_corrections
        if use_weights:
            errors = np.sqrt(errors**2 + calibration_errors**2)
        elif np.all(np.isfinite(calibration_errors)) and np.all(calibration_errors > 0):
            errors = calibration_errors.copy()
            use_weights = True
    else:
        observed = raw_observed.copy()

    def unpack(values: np.ndarray):
        parameter_map = {
            name: float(value)
            for name, value in zip(parameter_names, values)
        }
        zero_shift = parameter_map.pop("zero_shift", 0.0)
        return (
            _cell_from_system_parameters(
                parameter_map,
                crystal_system,
            ),
            float(zero_shift),
        )

    def calculated_positions(values: np.ndarray) -> np.ndarray:
        cell, zero_shift = unpack(values)
        positions = []
        for match in matches:
            calculated = two_theta_from_hkl(
                cell,
                tuple(int(value) for value in match["hkl"]),
                wavelength_angstrom,
            )
            if calculated is None or not np.isfinite(calculated):
                positions.append(np.nan)
            else:
                positions.append(calculated + zero_shift)
        return np.asarray(positions, dtype=float)

    def residual_function(values: np.ndarray) -> np.ndarray:
        calculated = calculated_positions(values)
        if np.any(~np.isfinite(calculated)):
            return np.full_like(observed, 1e3)
        residuals = observed - calculated
        return residuals / errors if use_weights else residuals

    result = least_squares(
        residual_function,
        np.asarray(initial_values, dtype=float),
        bounds=(
            np.asarray(lower_bounds, dtype=float),
            np.asarray(upper_bounds, dtype=float),
        ),
        max_nfev=20000,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )

    refined_cell, zero_shift = unpack(result.x)
    initial_parameter_values = np.asarray(initial_values, dtype=float)
    initial_calculated = calculated_positions(initial_parameter_values)
    refined_calculated = calculated_positions(result.x)
    initial_residuals = observed - initial_calculated
    refined_residuals = observed - refined_calculated

    parameter_errors = np.full(len(result.x), np.nan)
    degrees_of_freedom = max(1, len(observed) - len(result.x))
    if result.jac.size:
        try:
            covariance = np.linalg.inv(result.jac.T @ result.jac)
            residual_variance = float(
                np.sum(result.fun ** 2) / degrees_of_freedom
            )
            covariance *= residual_variance
            diagonal = np.diag(covariance)
            parameter_errors = np.sqrt(
                np.where(diagonal >= 0, diagonal, np.nan)
            )
        except np.linalg.LinAlgError:
            pass

    error_map = {
        name: (
            None
            if not np.isfinite(error)
            else float(error)
        )
        for name, error in zip(parameter_names, parameter_errors)
    }

    independent_error_values = {
        name: error_map.get(name)
        for name in ("a", "b", "c", "alpha", "beta", "gamma")
    }
    if crystal_system == "Cubic":
        independent_error_values.update(
            {"a": error_map.get("a"), "b": error_map.get("a"), "c": error_map.get("a")}
        )
    elif crystal_system in {"Tetragonal", "Hexagonal"}:
        independent_error_values.update(
            {"a": error_map.get("a"), "b": error_map.get("a"), "c": error_map.get("c")}
        )
    elif crystal_system == "Rhombohedral":
        independent_error_values.update(
            {
                "a": error_map.get("a"),
                "b": error_map.get("a"),
                "c": error_map.get("a"),
                "alpha": error_map.get("alpha"),
                "beta": error_map.get("alpha"),
                "gamma": error_map.get("alpha"),
            }
        )

    ss_res = float(np.sum(refined_residuals ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rmse = float(np.sqrt(np.mean(refined_residuals ** 2)))

    refined_matches = []
    for match, initial_calc, refined_calc, initial_delta, refined_delta in zip(
        matches,
        initial_calculated,
        refined_calculated,
        initial_residuals,
        refined_residuals,
    ):
        row = dict(match)
        row.update(
            {
                "raw_observed_2theta": float(match["observed_2theta"]),
                "instrument_position_correction_deg": float(
                    calibration_corrections[len(refined_matches)]
                ),
                "instrument_position_error_deg": float(
                    calibration_errors[len(refined_matches)]
                ),
                "corrected_observed_2theta": float(
                    observed[len(refined_matches)]
                ),
                "initial_calculated_2theta": float(initial_calc),
                "refined_calculated_2theta": float(refined_calc),
                "initial_delta": float(initial_delta),
                "refined_delta": float(refined_delta),
            }
        )
        refined_matches.append(row)

    tolerance_reference = max(
        max(abs(float(match["initial_delta"])) for match in matches),
        1e-9,
    )
    match_score = float(
        100.0 * np.mean(
            np.exp(
                -np.square(
                    refined_residuals / tolerance_reference
                )
            )
        )
    )

    return {
        "success": bool(result.success),
        "message": result.message,
        "crystal_system": crystal_system,
        "wavelength_angstrom": float(wavelength_angstrom),
        "refine_zero_shift": bool(refine_zero_shift),
        "instrument_position_correction_applied": bool(
            apply_instrument_position_correction
        ),
        "instrument_profile_fingerprint": (
            None if not instrument_profile else instrument_profile.get("fingerprint")
        ),
        "instrument_position_covariance_propagated": bool(
            apply_instrument_position_correction
            and use_weights
            and np.any(calibration_errors > 0)
        ),
        "initial_cell": {
            name: float(initial_cell[name])
            for name in ("a", "b", "c", "alpha", "beta", "gamma")
        },
        "refined_cell": refined_cell,
        "cell_errors": independent_error_values,
        "zero_shift_deg": float(zero_shift),
        "zero_shift_error_deg": error_map.get("zero_shift"),
        "match_count": len(matches),
        "weighted": use_weights,
        "r_squared": float(r_squared),
        "rmse_deg": rmse,
        "maximum_absolute_residual_deg": float(
            np.max(np.abs(refined_residuals))
        ),
        "match_score_percent": match_score,
        "matches": refined_matches,
    }
