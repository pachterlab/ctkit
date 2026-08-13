"""Shared fixtures: small synthetic volumes that exercise the real code paths."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import pytest


def make_volume(
    shape=(48, 48, 20),
    spacing=(0.7, 0.9, 2.5),
    flip=(-1, -1, 1),
    seed=0,
):
    """A synthetic CT-like volume: air, an "organ" sphere, a "tumor" inside it.

    Returns ``(image, mask)`` as :class:`nibabel.Nifti1Image` objects. The
    affine is built with sign flips so the storage orientation is not already
    canonical, which is what makes the reorientation step observable.
    """
    rng = np.random.default_rng(seed)
    grid = np.meshgrid(*[np.arange(size) for size in shape], indexing="ij")
    center = [size // 2 for size in shape]
    radius = min(shape[0], shape[1]) // 4

    distance = (
        (grid[0] - center[0]) ** 2
        + (grid[1] - center[1]) ** 2
        + ((grid[2] - center[2]) * 3) ** 2
    )
    organ = distance < radius ** 2
    tumor = distance < (radius // 2) ** 2

    data = np.full(shape, -1000.0)
    data[organ] = 60.0
    data[tumor] = 120.0
    data += rng.normal(0, 3, shape)

    mask = np.zeros(shape, dtype=np.uint8)
    mask[organ] = 1
    mask[tumor] = 2

    affine = np.diag([flip[0] * spacing[0], flip[1] * spacing[1], flip[2] * spacing[2], 1.0])
    return nib.Nifti1Image(data, affine), nib.Nifti1Image(mask, affine)


@pytest.fixture
def volume():
    return make_volume()


@pytest.fixture
def image(volume):
    from ctkit import RadiologyImage

    data, mask = volume
    return RadiologyImage(data, mask=mask, series_id="synthetic")


@pytest.fixture
def case_dir(tmp_path):
    """One case on disk: ``<dir>/case_00/{imaging,segmentation}.nii.gz``."""
    data, mask = make_volume()
    case = tmp_path / "case_00"
    case.mkdir()
    nib.save(data, str(case / "imaging.nii.gz"))
    nib.save(mask, str(case / "segmentation.nii.gz"))
    return case


@pytest.fixture
def cohort_dir(tmp_path):
    """A directory of five cases with differing shapes, plus one unusable case."""
    root = tmp_path / "cohort"
    root.mkdir()
    for index in range(5):
        data, mask = make_volume(
            shape=(40 + index * 4, 40 + index * 4, 16 + index),
            spacing=(0.7, 0.9, 2.0 + 0.1 * index),
            seed=index,
        )
        case = root / f"case_{index:02d}"
        case.mkdir()
        nib.save(data, str(case / "imaging.nii.gz"))
        nib.save(mask, str(case / "segmentation.nii.gz"))

    unusable = root / "case_bad"
    unusable.mkdir()
    nib.save(
        nib.Nifti1Image(np.zeros((32, 32, 3)), np.eye(4)),
        str(unusable / "imaging.nii.gz"),
    )
    return root


def has_radiomics() -> bool:
    try:
        import radiomics  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


requires_radiomics = pytest.mark.skipif(
    not has_radiomics(), reason="pyradiomics is not installed"
)

network = pytest.mark.skipif(
    os.environ.get("CTKIT_TEST_NETWORK") != "1",
    reason="set CTKIT_TEST_NETWORK=1 to run tests that contact TCIA",
)
