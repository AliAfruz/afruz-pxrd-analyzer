from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .models import Dataset


class ReferenceCardImportError(RuntimeError):
    pass


_REFERENCE_CARD_ROW = re.compile(
    r"""
    ^\s*
    (?P<two_theta>[+-]?\d+(?:\.\d+)?)
    \s+
    (?P<d_spacing>[+-]?\d+(?:\.\d+)?)
    \s+
    (?P<intensity><?\d+(?:\.\d+)?)
    \s+
    \(\s*
    (?P<h>[+-]?\d+)\s+
    (?P<k>[+-]?\d+)\s+
    (?P<l>[+-]?\d+)
    \s*\)
    (?:\s+
        (?P<theta>[+-]?\d+(?:\.\d+)?)
        \s+
        (?P<inverse_2d>[+-]?\d+(?:\.\d+)?)
        \s+
        (?P<two_pi_over_d>[+-]?\d+(?:\.\d+)?)
        (?:\s+(?P<n_squared>\S+))?
    )?
    \s*$
    """,
    re.VERBOSE,
)


def looks_like_reference_card_text(text: str) -> bool:
    """
    Detect a Jade/ICDD-style PDF reference-card *text export*.

    This does not parse binary PDF documents.
    """
    return bool(
        re.search(r"(?mi)^\s*PDF#[^:\r\n]+:", text)
        and re.search(r"(?mi)^\s*2-Theta\s+d(?:\(.*?\))?\s+I", text)
        and re.search(r"(?mi)\(\s*h\s+k\s+l\s*\)", text)
    )


def _metadata_float(line: str, key: str) -> float | None:
    match = re.search(
        rf"(?i){re.escape(key)}\s*=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)",
        line,
    )
    return None if match is None else float(match.group(1))


def _parse_reference_intensity(
    token: str,
) -> tuple[float, bool, float | None]:
    """
    Convert a reference-card intensity token to a plotting value.

    '<1' is an upper limit, not an exact value. It is plotted at half of the
    stated limit solely to make the reflection visible. The original token and
    upper limit remain stored in the reflection metadata.
    """
    token = token.strip()
    if token.startswith("<"):
        limit = float(token[1:])
        return limit / 2.0, True, limit
    return float(token), False, None


def parse_reference_card_text(
    text: str,
    source_path: str = "",
) -> Dataset:
    if not looks_like_reference_card_text(text):
        raise ReferenceCardImportError(
            "The text does not contain a recognized Jade/PDF reference-card table."
        )

    lines = text.splitlines()
    nonempty = [
        (index, line.strip())
        for index, line in enumerate(lines)
        if line.strip()
    ]

    header_match = None
    header_index = None
    for index, line in nonempty:
        match = re.match(
            r"^PDF#(?P<number>[^:]+):\s*(?P<details>.*)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            header_match = match
            header_index = index
            break
    if header_match is None:
        raise ReferenceCardImportError(
            "Could not read the PDF reference-card number."
        )

    pdf_number = header_match.group("number").strip()
    quality_details = header_match.group("details").strip()

    following = [
        line for index, line in nonempty if index > header_index
    ]
    compound_name = (
        following[0] if following else f"PDF {pdf_number}"
    )
    formula = following[1] if len(following) > 1 else ""

    radiation = ""
    wavelength = None
    filter_name = ""
    calibration_range = None
    rir = None
    crystal_system = ""
    space_group = ""
    space_group_number = None
    z_value = None
    cell = {}
    density_calculated = None
    volume = None
    strong_lines = ""
    references: list[str] = []
    notes: list[str] = []

    table_header_index = None
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if re.match(r"(?i)^2-Theta\s+d", line):
            table_header_index = index
            continue

        if line.lower().startswith("radiation="):
            match = re.search(
                r"(?i)Radiation\s*=\s*([^\t]+?)"
                r"(?=\s+Lambda\s*=|$)",
                line,
            )
            radiation = match.group(1).strip() if match else ""
            wavelength = _metadata_float(line, "Lambda")
            filter_match = re.search(
                r"(?i)\bFilter\s*=\s*([^\t]*)",
                line,
            )
            if filter_match:
                filter_name = filter_match.group(1).strip()
            continue

        if line.lower().startswith("calibration="):
            range_match = re.search(
                r"(?i)\b2T\s*=\s*"
                r"([+-]?\d+(?:\.\d+)?)\s*-\s*"
                r"([+-]?\d+(?:\.\d+)?)",
                line,
            )
            if range_match:
                calibration_range = [
                    float(range_match.group(1)),
                    float(range_match.group(2)),
                ]
            rir = _metadata_float(line, "I/Ic(RIR)")
            continue

        crystal_match = re.match(
            r"^(?P<system>Cubic|Tetragonal|Orthorhombic|Hexagonal|"
            r"Trigonal|Rhombohedral|Monoclinic|Triclinic)"
            r"\s*,\s*(?P<group>.*?)"
            r"(?:\s+\((?P<number>\d+)\))?"
            r"\s+Z\s*=\s*(?P<z>\d+)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if crystal_match:
            crystal_system = crystal_match.group("system").title()
            space_group = crystal_match.group("group").strip()
            if crystal_match.group("number"):
                space_group_number = int(
                    crystal_match.group("number")
                )
            z_value = int(crystal_match.group("z"))
            continue

        if line.upper().startswith("CELL:"):
            cell_match = re.search(
                r"(?i)CELL:\s*"
                r"([+-]?\d+(?:\.\d+)?)\s*x\s*"
                r"([+-]?\d+(?:\.\d+)?)\s*x\s*"
                r"([+-]?\d+(?:\.\d+)?)\s*"
                r"<\s*"
                r"([+-]?\d+(?:\.\d+)?)\s*x\s*"
                r"([+-]?\d+(?:\.\d+)?)\s*x\s*"
                r"([+-]?\d+(?:\.\d+)?)\s*>",
                line,
            )
            if cell_match:
                values = [
                    float(value)
                    for value in cell_match.groups()
                ]
                cell = {
                    "a": values[0],
                    "b": values[1],
                    "c": values[2],
                    "alpha": values[3],
                    "beta": values[4],
                    "gamma": values[5],
                }
            continue

        if line.lower().startswith("density(c)="):
            density_calculated = _metadata_float(
                line,
                "Density(c)",
            )
            volume = _metadata_float(line, "Vol")
            continue

        if line.lower().startswith("strong lines:"):
            strong_lines = line.split(":", 1)[1].strip()
            continue

        if line.lower().startswith("ref:"):
            references.append(line[4:].strip())
            continue

        if line.lower().startswith("note:"):
            notes.append(line[5:].strip())
            continue

    if table_header_index is None:
        raise ReferenceCardImportError(
            "Could not locate the reflection-table header."
        )

    reflections = []
    for raw_line in lines[table_header_index + 1:]:
        match = _REFERENCE_CARD_ROW.match(raw_line)
        if not match:
            continue

        intensity_text = match.group("intensity")
        (
            intensity_value,
            is_upper_limit,
            intensity_upper_limit,
        ) = _parse_reference_intensity(intensity_text)

        h = int(match.group("h"))
        k = int(match.group("k"))
        l = int(match.group("l"))
        reflections.append(
            {
                "two_theta": float(match.group("two_theta")),
                "d_spacing_angstrom": float(
                    match.group("d_spacing")
                ),
                "intensity": float(intensity_value),
                "intensity_text": intensity_text,
                "intensity_is_upper_limit": bool(is_upper_limit),
                "intensity_upper_limit": intensity_upper_limit,
                "h": h,
                "k": k,
                "l": l,
                "hkl_label": f"({h} {k} {l})",
                "theta": (
                    None
                    if match.group("theta") is None
                    else float(match.group("theta"))
                ),
                "inverse_2d": (
                    None
                    if match.group("inverse_2d") is None
                    else float(match.group("inverse_2d"))
                ),
                "two_pi_over_d": (
                    None
                    if match.group("two_pi_over_d") is None
                    else float(match.group("two_pi_over_d"))
                ),
                "n_squared": match.group("n_squared") or "",
            }
        )

    if len(reflections) < 3:
        raise ReferenceCardImportError(
            "Fewer than three reflection rows were parsed."
        )

    reflections.sort(key=lambda row: row["two_theta"])
    x = np.asarray(
        [row["two_theta"] for row in reflections],
        dtype=float,
    )
    y = np.asarray(
        [row["intensity"] for row in reflections],
        dtype=float,
    )

    short_formula = formula.strip() or compound_name
    dataset_name = f"PDF {pdf_number} — {short_formula}"
    source = (
        str(Path(source_path).resolve())
        if source_path
        else ""
    )

    dataset = Dataset(
        name=dataset_name,
        x=x,
        y_raw=y,
        source_path=source,
        metadata={
            "import_format": (
                Path(source_path).suffix.lower()
                if source_path
                else "reference-card-text"
            ),
            "source_format": "Jade/PDF reference-card text",
            "analysis_role": "reference_pattern",
            "plot_style": "sticks",
            "pdf_number": pdf_number,
            "quality_details": quality_details,
            "compound_name": compound_name,
            "formula": formula,
            "radiation": radiation,
            "wavelength_k_alpha1": wavelength,
            "filter": filter_name,
            "calibration_two_theta_range": calibration_range,
            "rir": rir,
            "crystal_system": crystal_system,
            "space_group": space_group,
            "space_group_number": space_group_number,
            "z": z_value,
            "cell": cell,
            "density_calculated": density_calculated,
            "volume_angstrom3": volume,
            "strong_lines": strong_lines,
            "references": references,
            "notes": notes,
            "reflections": reflections,
            "upper_limit_plotting_rule": (
                "An intensity token '<x' is plotted at x/2 only for "
                "visibility; the original token and upper limit are preserved."
            ),
            "points": len(reflections),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "x_axis": "2Theta",
            "x_unit": "degree",
            "intensity_unit": "relative intensity",
        },
    )
    dataset.validate()
    return dataset


def load_reference_card_pattern(
    path: str | Path,
) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise ReferenceCardImportError(
            f"File does not exist: {path}"
        )
    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise ReferenceCardImportError(
            f"Could not read reference-card text file: {exc}"
        ) from exc
    return parse_reference_card_text(
        text,
        source_path=str(path),
    )
