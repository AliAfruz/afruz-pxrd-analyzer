from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTEST_QT_API", "pyside6")

import pandas as pd
import pytest

from afruz_pxrd.crystallography import load_cif


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA = PROJECT_ROOT / "Afruz_PXRD_Synthetic_Test_Set"
STATISTICAL_DATA = PROJECT_ROOT / "Afruz_PXRD_Statistical_Validation_Set"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def synthetic_data_dir() -> Path:
    return SYNTHETIC_DATA


@pytest.fixture(scope="session")
def statistical_data_dir() -> Path:
    return STATISTICAL_DATA


@pytest.fixture(scope="session")
def nacl_structure():
    return load_cif(STATISTICAL_DATA / "NaCl_validation_P1.cif")


@pytest.fixture(scope="session")
def cscl_structure():
    return load_cif(STATISTICAL_DATA / "CsCl_validation_P1.cif")


@pytest.fixture(scope="session")
def nacl_raw_frame():
    return pd.read_csv(
        STATISTICAL_DATA / "01_NaCl_raw_counts_with_sigma.txt",
        comment="#",
        sep="\t",
    )

@pytest.fixture(scope="session")
def nacl_normalized_frame():
    return pd.read_csv(
        STATISTICAL_DATA / "02_NaCl_normalized_100_with_sigma.txt",
        comment="#",
        sep="\t",
    )


@pytest.fixture(scope="session")
def mixed_raw_frame():
    return pd.read_csv(
        STATISTICAL_DATA / "04_NaCl_CsCl_mixed_raw_with_sigma.txt",
        comment="#",
        sep="\t",
    )


@pytest.fixture(scope="session")
def generating_profile_frame():
    return pd.read_csv(
        STATISTICAL_DATA / "03_NaCl_generating_profile_reference.txt",
        comment="#",
        sep="\t",
    )
