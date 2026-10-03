from __future__ import annotations

from pathlib import Path
import csv
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .models import Dataset
from .text_export import write_columns_txt
from .version import APP_NAME, APP_VERSION
from .reference_card import (
    ReferenceCardImportError,
    load_reference_card_pattern,
    looks_like_reference_card_text,
    parse_reference_card_text,
)

TEXT_EXTENSIONS = {".xy", ".csv", ".txt", ".dat"}
RD_EXTENSIONS = {".rd"}
REFERENCE_CARD_EXTENSIONS = {
    ".pdfcard",
    ".jcpds",
    ".jade",
    ".card",
    ".ref",
}
XRDML_EXTENSIONS = {".xrdml"}
SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | REFERENCE_CARD_EXTENSIONS
    | XRDML_EXTENSIONS
    | RD_EXTENSIONS
)


class DataImportError(RuntimeError):
    pass


def _guess_delimiter(sample: str) -> str | None:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        return dialect.delimiter
    except csv.Error:
        return None


def _local_name(tag: str) -> str:
    """Return an XML tag name without its namespace."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _local_name(child.tag) == name:
            return child
    return None


def _first_descendant(element: ET.Element, name: str) -> ET.Element | None:
    for child in element.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [
        child for child in element.iter()
        if _local_name(child.tag) == name
    ]


def _element_text(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else element.text.strip()


def _float_text(element: ET.Element | None) -> float | None:
    text = _element_text(element)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _numeric_array(text: str) -> np.ndarray:
    normalized = " ".join(text.replace(",", " ").split())
    if not normalized:
        return np.asarray([], dtype=float)
    return np.fromstring(normalized, sep=" ", dtype=float)


def _convert_position_unit(values: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    normalized = (unit or "").strip().lower()
    if normalized in {"rad", "radian", "radians"}:
        return np.degrees(values), "degree"
    return values, unit or "degree"


def _position_values(
    position_element: ET.Element,
    point_count: int,
) -> tuple[np.ndarray | None, str]:
    unit = position_element.attrib.get("unit", "degree")

    list_element = _first_child(position_element, "listPositions")
    if list_element is not None:
        values = _numeric_array(_element_text(list_element))
        if len(values) == point_count:
            return _convert_position_unit(values, unit)

    start = _float_text(_first_child(position_element, "startPosition"))
    end = _float_text(_first_child(position_element, "endPosition"))
    if start is not None and end is not None:
        values = np.linspace(start, end, point_count)
        return _convert_position_unit(values, unit)

    common = _first_child(position_element, "commonPosition")
    if common is not None:
        values = _numeric_array(_element_text(common))
        if len(values) == point_count:
            return _convert_position_unit(values, unit)

    return None, unit


def _choose_x_axis(
    data_points: ET.Element,
    point_count: int,
) -> tuple[np.ndarray, str, str]:
    candidates = []
    for position in _children(data_points, "positions"):
        values, normalized_unit = _position_values(position, point_count)
        if values is None:
            continue
        axis = position.attrib.get("axis", "unknown")
        axis_key = axis.lower().replace("-", "").replace("_", "")
        priority = 0
        if "2theta" in axis_key:
            priority = 3
        elif axis_key in {"twotheta", "theta2theta"}:
            priority = 3
        elif "omega" in axis_key or "theta" in axis_key:
            priority = 2
        else:
            priority = 1
        candidates.append((priority, axis, values, normalized_unit))

    if not candidates:
        raise DataImportError(
            "No varying scan-position array matching the intensity length was found."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, axis, values, unit = candidates[0]
    return values, axis, unit


def _scan_intensity_data(
    data_points: ET.Element,
) -> tuple[np.ndarray, str, str]:
    for tag_name in ("intensities", "counts"):
        element = _first_child(data_points, tag_name)
        if element is None:
            continue
        values = _numeric_array(_element_text(element))
        if values.size:
            unit = element.attrib.get(
                "unit",
                "counts" if tag_name == "counts" else "intensity",
            )
            return values, tag_name, unit
    raise DataImportError("The scan contains neither intensities nor counts.")


def _counting_time_metadata(
    data_points: ET.Element,
    point_count: int,
) -> dict:
    metadata = {}

    common = _float_text(_first_child(data_points, "commonCountingTime"))
    if common is not None:
        metadata["counting_time_s"] = common
        metadata["counting_time_mode"] = "common"
        return metadata

    times_element = _first_child(data_points, "countingTimes")
    if times_element is not None:
        times = _numeric_array(_element_text(times_element))
        if len(times) == point_count:
            metadata["counting_time_mode"] = "per_point"
            metadata["counting_time_min_s"] = float(np.min(times))
            metadata["counting_time_max_s"] = float(np.max(times))
            metadata["counting_times_s"] = times.tolist()
    return metadata


def _measurement_metadata(
    root: ET.Element,
    measurement: ET.Element,
) -> dict:
    metadata = {}

    sample = _first_descendant(root, "sample")
    if sample is not None:
        sample_id = _element_text(_first_descendant(sample, "id"))
        sample_name = _element_text(_first_descendant(sample, "name"))
        if sample_id:
            metadata["sample_id"] = sample_id
        if sample_name:
            metadata["sample_name"] = sample_name

    used_wavelength = _first_descendant(measurement, "usedWavelength")
    if used_wavelength is not None:
        wavelength_tags = {
            "kAlpha1": "wavelength_k_alpha1",
            "kAlpha2": "wavelength_k_alpha2",
            "kBeta": "wavelength_k_beta",
            "ratioKAlpha2KAlpha1": "k_alpha2_to_k_alpha1_ratio",
        }
        for xml_name, metadata_name in wavelength_tags.items():
            value = _float_text(_first_child(used_wavelength, xml_name))
            if value is not None:
                metadata[metadata_name] = value

    xray_tube = _first_descendant(measurement, "xRayTube")
    if xray_tube is not None:
        tube_name = xray_tube.attrib.get("name")
        if tube_name:
            metadata["xray_tube"] = tube_name
        tension = _float_text(_first_descendant(xray_tube, "tension"))
        current = _float_text(_first_descendant(xray_tube, "current"))
        if tension is not None:
            metadata["tube_tension_kv"] = tension
        if current is not None:
            metadata["tube_current_ma"] = current

    measurement_type = measurement.attrib.get("measurementType")
    if measurement_type:
        metadata["measurement_type"] = measurement_type

    return metadata


def _scan_metadata(
    scan: ET.Element,
    scan_index: int,
    scan_count: int,
) -> dict:
    metadata = {
        "scan_index": scan_index,
        "scan_count_in_file": scan_count,
    }
    for attribute in ("scanAxis", "mode", "status"):
        value = scan.attrib.get(attribute)
        if value:
            metadata[attribute] = value

    header = _first_child(scan, "header")
    if header is not None:
        start = _element_text(_first_child(header, "startTimeStamp"))
        end = _element_text(_first_child(header, "endTimeStamp"))
        if start:
            metadata["start_timestamp"] = start
        if end:
            metadata["end_timestamp"] = end

    non_ambient = _first_child(scan, "nonAmbientPoints")
    if non_ambient is not None:
        condition_type = non_ambient.attrib.get("type")
        values_element = _first_child(non_ambient, "nonAmbientValues")
        values = _numeric_array(_element_text(values_element))
        if condition_type:
            metadata["non_ambient_type"] = condition_type
        if values.size:
            metadata["non_ambient_values"] = values.tolist()

    return metadata


def load_xrdml_patterns(path: str | Path) -> list[Dataset]:
    """
    Load all one-dimensional scans from a Malvern Panalytical XRDML file.

    Each scan is returned as an independent Dataset. Repeated scans are not
    silently averaged, preserving the original measurement records.
    """
    path = Path(path)
    if not path.exists():
        raise DataImportError(f"File does not exist: {path}")
    if path.suffix.lower() not in XRDML_EXTENSIONS:
        raise DataImportError(f"Not an XRDML file: {path.name}")

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise DataImportError(f"Invalid XRDML/XML structure: {exc}") from exc
    except OSError as exc:
        raise DataImportError(f"Could not read XRDML file: {exc}") from exc

    root = tree.getroot()
    measurements = _descendants(root, "xrdMeasurement")
    if not measurements and _local_name(root.tag) == "xrdMeasurement":
        measurements = [root]
    if not measurements:
        raise DataImportError("No xrdMeasurement element was found.")

    parsed_records = []
    all_scans = []
    for measurement_index, measurement in enumerate(measurements, start=1):
        scans = _children(measurement, "scan")
        if not scans:
            scans = [
                element for element in measurement.iter()
                if _local_name(element.tag) == "scan"
            ]
        for scan in scans:
            all_scans.append((measurement_index, measurement, scan))

    if not all_scans:
        raise DataImportError("No scan elements were found in the XRDML measurement.")

    scan_count = len(all_scans)
    for global_index, (measurement_index, measurement, scan) in enumerate(
        all_scans,
        start=1,
    ):
        data_points = _first_child(scan, "dataPoints")
        if data_points is None:
            data_points = _first_descendant(scan, "dataPoints")
        if data_points is None:
            continue

        try:
            y, intensity_tag, intensity_unit = _scan_intensity_data(data_points)
            x, x_axis, x_unit = _choose_x_axis(data_points, len(y))
        except DataImportError:
            # Keep trying other scans, then report a useful aggregate error.
            continue

        metadata = {
            "import_format": ".xrdml",
            "source_format": "XRDML",
            "x_axis": x_axis,
            "x_unit": x_unit,
            "intensity_element": intensity_tag,
            "intensity_unit": intensity_unit,
            "measurement_index": measurement_index,
            "points": int(len(x)),
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
        }
        metadata.update(_measurement_metadata(root, measurement))
        metadata.update(_scan_metadata(scan, global_index, scan_count))
        metadata.update(_counting_time_metadata(data_points, len(y)))

        scan_axis = scan.attrib.get("scanAxis") or x_axis
        if scan_count == 1:
            dataset_name = path.stem
        else:
            dataset_name = f"{path.stem} — Scan {global_index}"
            if scan_axis:
                dataset_name += f" ({scan_axis})"

        dataset = Dataset(
            name=dataset_name,
            x=np.asarray(x, dtype=float),
            y_raw=np.asarray(y, dtype=float),
            source_path=str(path.resolve()),
            metadata=metadata,
        )
        try:
            dataset.validate()
        except ValueError as exc:
            raise DataImportError(
                f"Scan {global_index} contains invalid coordinate data: {exc}"
            ) from exc
        parsed_records.append(dataset)

    if not parsed_records:
        raise DataImportError(
            "XRDML scans were found, but no supported one-dimensional pattern "
            "contained matching positions and intensities/counts."
        )

    return parsed_records


def _looks_like_ordinal_column(values: np.ndarray) -> bool:
    """Return True for 1, 2, 3 ... row-number columns."""
    if values.size < 3:
        return False
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-6):
        return False
    diffs = np.diff(rounded)
    return bool(np.all(diffs == 1) and rounded[0] in {0, 1})


def _looks_like_monotonic_x(values: np.ndarray) -> bool:
    if values.size < 3 or not np.all(np.isfinite(values)):
        return False
    diffs = np.diff(values)
    if not np.all(diffs > 0):
        return False
    # PXRD 2θ values normally change smoothly.  Keep this permissive so
    # sparse exported peak/intensity tables also load cleanly.
    median_step = float(np.median(diffs))
    if median_step <= 0:
        return False
    return bool(np.max(np.abs(diffs - median_step)) <= max(1.0, 20.0 * median_step))


def _select_text_pattern_columns(numeric: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    """Choose 2θ and intensity columns from generic numeric tables.

    Many instrument/export tables contain a leading row-number column, for
    example ``No.  Pos. [°2Th.]  Iobs [cts]``.  Older Afruz versions took
    the first two numeric columns and therefore read No. as the X axis.
    This selector first detects and skips such ordinal columns, then favors
    a monotonically increasing 2θ-like column followed by the next numeric
    intensity column.
    """
    if numeric.shape[1] < 2:
        raise ValueError("At least two numeric columns are required.")

    candidate = numeric.dropna()
    if candidate.shape[0] < 3:
        raise ValueError("At least three complete numeric rows are required.")

    arrays = [candidate.iloc[:, index].to_numpy(dtype=float) for index in range(candidate.shape[1])]

    # Common exported pattern table: row number, 2θ, intensity.
    if candidate.shape[1] >= 3 and _looks_like_ordinal_column(arrays[0]):
        if _looks_like_monotonic_x(arrays[1]):
            return (
                candidate.iloc[:, 1],
                candidate.iloc[:, 2],
                {
                    "text_column_rule": "skipped leading ordinal column",
                    "x_column_index": 1,
                    "intensity_column_index": 2,
                },
            )

    for x_index, x_values in enumerate(arrays):
        if not _looks_like_monotonic_x(x_values):
            continue
        for y_index in range(candidate.shape[1]):
            if y_index == x_index:
                continue
            return (
                candidate.iloc[:, x_index],
                candidate.iloc[:, y_index],
                {
                    "text_column_rule": "monotonic 2theta-like x column",
                    "x_column_index": x_index,
                    "intensity_column_index": y_index,
                },
            )

    # Conservative fallback for older two-column XY data.
    return (
        candidate.iloc[:, 0],
        candidate.iloc[:, 1],
        {
            "text_column_rule": "first two numeric columns",
            "x_column_index": 0,
            "intensity_column_index": 1,
        },
    )




def _text_intensity_unit(text: str) -> tuple[str, str]:
    """Infer intensity units conservatively from text headers/comments."""
    header = "\n".join(text.splitlines()[:80]).lower()
    count_tokens = (
        "intensity_counts",
        "intensity count",
        "counts",
        "[cts]",
        "(cts)",
        " iobs [cts]",
    )
    rate_tokens = ("counts/s", "counts per second", "cps", "count rate")
    if any(token in header for token in rate_tokens):
        return "counts/s", "text header"
    if any(token in header for token in count_tokens):
        return "counts", "text header"
    if "a.u." in header or "arbitrary unit" in header or "normalized" in header:
        return "a.u.", "text header"
    return "a.u.", "not specified"

def load_text_pattern(path: str | Path) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise DataImportError(f"File does not exist: {path}")
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        raise DataImportError(
            f"Supported text formats are {sorted(TEXT_EXTENSIONS)}. "
            f"Format {path.suffix or '<none>'} requires a dedicated parser."
        )

    text = path.read_text(encoding="utf-8", errors="ignore")
    if looks_like_reference_card_text(text):
        try:
            return parse_reference_card_text(
                text,
                source_path=str(path),
            )
        except ReferenceCardImportError as exc:
            raise DataImportError(str(exc)) from exc

    sample = "\n".join(text.splitlines()[:30])
    delimiter = _guess_delimiter(sample)
    intensity_unit, intensity_unit_source = _text_intensity_unit(text)

    parse_attempts = []
    for sep in ([delimiter] if delimiter else []) + [None, r"\s+", ",", ";", "\t"]:
        if sep in parse_attempts:
            continue
        parse_attempts.append(sep)
        try:
            frame = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                comment="#",
                header=None,
                on_bad_lines="skip",
            )
            numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            numeric = numeric.dropna(axis=1, how="all")
            if numeric.shape[1] < 2:
                continue
            x_series, y_series, column_metadata = _select_text_pattern_columns(numeric)
            x = x_series.to_numpy(dtype=float)
            y = y_series.to_numpy(dtype=float)
            ds = Dataset(name=path.stem, x=x, y_raw=y, source_path=str(path.resolve()))
            ds.validate()
            ds.metadata.update(
                {
                    "points": int(len(ds.x)),
                    "x_min": float(ds.x.min()),
                    "x_max": float(ds.x.max()),
                    "import_format": path.suffix.lower(),
                    "source_format": "generic text table",
                    "x_axis": "2Theta",
                    "x_unit": "degree",
                    "intensity_unit": intensity_unit,
                    "intensity_unit_source": intensity_unit_source,
                    "count_statistics_detected": intensity_unit in {"counts", "counts/s"},
                }
            )
            ds.metadata.update(column_metadata)
            return ds
        except Exception:
            continue

    raise DataImportError(
        "Could not identify numeric 2θ and intensity columns. "
        "Supported text examples: two-column XY or No./Pos./Iobs tables."
    )


def _unpack_little_double(data: bytes, offset: int, label: str) -> float:
    try:
        return struct.unpack_from("<d", data, offset)[0]
    except struct.error as exc:
        raise DataImportError(f"RD header is too short for {label}.") from exc


def load_rd_pattern(path: str | Path) -> Dataset:
    """Load Rigaku/SmartLab-style V3RD binary RD scan files.

    The supported V3RD layout stores Cu wavelength metadata in the header,
    2θ bin edges near offsets 214/222/230, and unsigned 16-bit counts from
    byte 250 onward.  Positions are returned as bin centers, which matches
    exported tables where a 4.900° start edge and 0.050° step become the
    first point at 4.925° 2θ.
    """
    path = Path(path)
    if not path.exists():
        raise DataImportError(f"File does not exist: {path}")
    if path.suffix.lower() not in RD_EXTENSIONS:
        raise DataImportError(f"Not an RD file: {path.name}")

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DataImportError(f"Could not read RD file: {exc}") from exc

    magic = data[:4].decode("ascii", errors="ignore")
    if magic != "V3RD":
        raise DataImportError(
            f"Unsupported RD binary header {magic!r}. "
            "This importer currently supports V3RD scans only."
        )

    data_offset = 250
    if len(data) <= data_offset + 6:
        raise DataImportError("RD file is too small to contain a V3RD scan.")

    step = _unpack_little_double(data, 214, "2θ step")
    start_edge = _unpack_little_double(data, 222, "2θ start edge")
    end_edge = _unpack_little_double(data, 230, "2θ end edge")
    wavelength_k_alpha1 = _unpack_little_double(data, 94, "Kα1 wavelength")
    wavelength_k_alpha2 = _unpack_little_double(data, 102, "Kα2 wavelength")
    k_alpha_ratio = _unpack_little_double(data, 110, "Kα2/Kα1 ratio")

    if not np.isfinite(step) or step <= 0:
        raise DataImportError("RD file contains an invalid 2θ step.")
    if not np.isfinite(start_edge) or not np.isfinite(end_edge) or end_edge <= start_edge:
        raise DataImportError("RD file contains invalid 2θ scan limits.")

    available_bytes = len(data) - data_offset
    point_count = available_bytes // 2
    if point_count < 3:
        raise DataImportError("RD file contains fewer than three intensity points.")

    expected_points = int(round((end_edge - start_edge) / step))
    if expected_points > 2 and abs(expected_points - point_count) <= 2:
        point_count = min(point_count, expected_points)

    counts = np.frombuffer(data[data_offset:data_offset + 2 * point_count], dtype="<u2").astype(float)
    x = start_edge + (0.5 * step) + np.arange(point_count, dtype=float) * step

    ds = Dataset(name=path.stem, x=x, y_raw=counts, source_path=str(path.resolve()))
    try:
        ds.validate()
    except ValueError as exc:
        raise DataImportError(f"RD scan contains invalid coordinate data: {exc}") from exc

    metadata = {
        "points": int(len(ds.x)),
        "x_min": float(ds.x.min()),
        "x_max": float(ds.x.max()),
        "import_format": ".rd",
        "source_format": "V3RD binary RD",
        "x_axis": "2Theta",
        "x_unit": "degree",
        "intensity_unit": "counts",
        "rd_magic": magic,
        "rd_data_offset_bytes": data_offset,
        "rd_start_edge_deg": float(start_edge),
        "rd_end_edge_deg": float(end_edge),
        "rd_step_deg": float(step),
        "rd_parser_scope": "validated for V3RD files matching the supplied sample layout",
    }
    if np.isfinite(wavelength_k_alpha1) and 0.1 < wavelength_k_alpha1 < 10:
        metadata["wavelength_k_alpha1"] = float(wavelength_k_alpha1)
    if np.isfinite(wavelength_k_alpha2) and 0.1 < wavelength_k_alpha2 < 10:
        metadata["wavelength_k_alpha2"] = float(wavelength_k_alpha2)
    if np.isfinite(k_alpha_ratio) and 0 <= k_alpha_ratio <= 1:
        metadata["k_alpha2_to_k_alpha1_ratio"] = float(k_alpha_ratio)
    ds.metadata.update(metadata)
    return ds


def load_patterns(path: str | Path) -> list[Dataset]:
    """Dispatch a supported source file to the correct importer."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension in XRDML_EXTENSIONS:
        return load_xrdml_patterns(path)
    if extension in RD_EXTENSIONS:
        return [load_rd_pattern(path)]
    if extension in REFERENCE_CARD_EXTENSIONS:
        try:
            return [load_reference_card_pattern(path)]
        except ReferenceCardImportError as exc:
            raise DataImportError(str(exc)) from exc
    if extension in TEXT_EXTENSIONS:
        return [load_text_pattern(path)]
    raise DataImportError(
        f"Unsupported format {extension or '<none>'}. "
        f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )


def _dataset_export_frame(dataset: Dataset) -> pd.DataFrame:
    reflections = dataset.metadata.get("reflections")
    if (
        dataset.metadata.get("analysis_role")
        == "reference_pattern"
        and isinstance(reflections, list)
        and reflections
    ):
        return pd.DataFrame(reflections)

    columns = {
        "two_theta_deg": dataset.x,
        "intensity_raw": dataset.y_raw,
    }
    # Do not repeat raw values under a misleading "processed" heading when no
    # treatment has been applied.  The optional third column is the actual
    # analysis-ready profile.
    if dataset.y_processed is not None:
        columns["intensity_treated"] = dataset.y_processed
    return pd.DataFrame(columns)




def export_dataset_txt(
    dataset: Dataset,
    path: str | Path,
) -> dict:
    """Export one dataset as a minimal, clean tab-delimited UTF-8 TXT table."""
    frame = _dataset_export_frame(dataset)
    return write_columns_txt(
        path,
        {str(column): frame[column].tolist() for column in frame.columns},
        title=f"PXRD dataset — {dataset.name}",
        metadata={
            "application_name": APP_NAME,
            "application_version": APP_VERSION,
            "dataset_uid": dataset.uid,
            "dataset_name": dataset.name,
            "source_path": dataset.source_path or "Not recorded",
            "point_count": len(dataset.x),
        },
    )

def export_dataset_csv(
    dataset: Dataset,
    path: str | Path,
) -> None:
    _dataset_export_frame(dataset).to_csv(path, index=False)


def export_dataset_excel(
    dataset: Dataset,
    path: str | Path,
) -> None:
    """Export a standalone dataset through the clean styled workbook engine."""
    from .export_engine import build_clean_export_package, write_excel_workbook

    package = build_clean_export_package(
        {
            "project_name": dataset.name,
            "project_path": "Standalone dataset export",
            "application_version": APP_VERSION,
            "datasets": [
                {
                    "uid": dataset.uid,
                    "name": dataset.name,
                    "source_path": dataset.source_path,
                    "x": dataset.x,
                    "y_raw": dataset.y_raw,
                    "y_processed": dataset.y_processed,
                    "metadata": dataset.metadata,
                    "analyses": {},
                }
            ],
        },
        include_content={"raw", "treated"},
    )
    write_excel_workbook(path, package)
