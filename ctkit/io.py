"""Reading and writing images.

Everything in this package is held in memory as a :class:`nibabel.Nifti1Image`.
This module is the only place that touches the filesystem for image data, and
the only place that converts between NIfTI, SimpleITK and NumPy.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import Any, Optional, Sequence, Union

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .validation import OUTPUT_FORMATS

logger = logging.getLogger(__name__)

NIFTI_SUFFIXES = (".nii", ".nii.gz")
ImageLike = Union[str, "os.PathLike[str]", nib.Nifti1Image, sitk.Image, np.ndarray]


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load_image(
    source: ImageLike,
    affine: Optional[np.ndarray] = None,
    spacing: Optional[Sequence[float]] = None,
) -> nib.Nifti1Image:
    """Load anything image-shaped into a :class:`nibabel.Nifti1Image`.

    Accepts a path to a NIfTI file, a ``.npy`` array, a directory of DICOM
    slices, a zip of DICOM slices, a single DICOM file, or an already-loaded
    ``Nifti1Image`` / ``SimpleITK.Image`` / :class:`numpy.ndarray`.

    ``affine`` or ``spacing`` are only used when the source carries no spatial
    metadata of its own (a bare array).
    """
    if isinstance(source, nib.Nifti1Image):
        return source

    if isinstance(source, sitk.Image):
        return sitk_to_nifti(source)

    if isinstance(source, np.ndarray):
        return array_to_nifti(source, affine=affine, spacing=spacing)

    if isinstance(source, (str, os.PathLike)):
        return _load_path(str(source), affine=affine, spacing=spacing)

    raise TypeError(
        f"Cannot load an image from {type(source).__name__}. Expected a path, "
        "a nibabel.Nifti1Image, a SimpleITK.Image, or a numpy.ndarray."
    )


def _load_path(
    path: str,
    affine: Optional[np.ndarray] = None,
    spacing: Optional[Sequence[float]] = None,
) -> nib.Nifti1Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such image: {path}")

    if os.path.isdir(path):
        return dicom_to_nifti(path)

    lower = path.lower()
    if lower.endswith(NIFTI_SUFFIXES):
        image = nib.load(path)
        if not isinstance(image, nib.Nifti1Image):  # .mgz and friends
            image = nib.Nifti1Image(
                np.asanyarray(image.dataobj), image.affine, None
            )
        return image
    if lower.endswith(".npy"):
        return array_to_nifti(np.load(path), affine=affine, spacing=spacing)
    if lower.endswith(".npz"):
        with np.load(path) as bundle:
            key = "image" if "image" in bundle else list(bundle)[0]
            return array_to_nifti(bundle[key], affine=affine, spacing=spacing)
    if lower.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as archive:
                archive.extractall(tmp)
            return dicom_to_nifti(tmp)
    if lower.endswith((".dcm", ".ima")) or _looks_like_dicom(path):
        return dicom_to_nifti(os.path.dirname(path) or ".")

    # Last resort: let nibabel try (handles .mgz, .hdr/.img, .mnc, ...).
    image = nib.load(path)
    return nib.Nifti1Image(np.asanyarray(image.dataobj), image.affine, None)


def is_dicom_directory(path: str) -> bool:
    """True for a directory of DICOM slices, i.e. one series rather than a cohort.

    Only the files directly inside `path` are examined, which is what separates
    a single series from a directory of per-case subdirectories.
    """
    try:
        entries = sorted(os.listdir(path))
    except OSError:
        return False
    for name in entries[:50]:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate) and (
            name.lower().endswith((".dcm", ".ima")) or _looks_like_dicom(candidate)
        ):
            return True
    return False


def _looks_like_dicom(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def array_to_nifti(
    array: np.ndarray,
    affine: Optional[np.ndarray] = None,
    spacing: Optional[Sequence[float]] = None,
) -> nib.Nifti1Image:
    """Wrap a raw array, building an affine from `spacing` when given.

    A bare array has no world coordinates. We default to an identity affine
    (1 mm isotropic, RAS), which is fine for shape-based work but means
    physical-space steps such as resampling are meaningless until you supply
    real spacing.
    """
    if affine is None:
        affine = np.eye(4)
        if spacing is not None:
            affine[:3, :3] = np.diag([float(s) for s in list(spacing)[:3]])
    return nib.Nifti1Image(np.asarray(array), np.asarray(affine, dtype=float))


# ----------------------------------------------------------------------
# DICOM
# ----------------------------------------------------------------------
def dicom_to_nifti(
    dicom_dir: str,
    prefer: str = "auto",
) -> nib.Nifti1Image:
    """Convert a directory of DICOM slices into a NIfTI volume, in memory.

    Uses ``dcm2niix`` when it is on the PATH (it handles more scanner quirks),
    otherwise SimpleITK's series reader. Any files produced by ``dcm2niix``
    live in a temporary directory that is removed before this returns.
    """
    if prefer not in ("auto", "dcm2niix", "sitk"):
        raise ValueError(f"prefer must be 'auto', 'dcm2niix' or 'sitk', got {prefer!r}")

    use_dcm2niix = prefer == "dcm2niix" or (
        prefer == "auto" and shutil.which("dcm2niix") is not None
    )
    if use_dcm2niix:
        try:
            return _dicom_to_nifti_dcm2niix(dicom_dir)
        except Exception as error:  # noqa: BLE001 - fall through to SimpleITK
            if prefer == "dcm2niix":
                raise
            logger.debug("dcm2niix failed on %s (%s); falling back to SimpleITK",
                         dicom_dir, error)
    return _dicom_to_nifti_sitk(dicom_dir)


def _dicom_to_nifti_dcm2niix(dicom_dir: str) -> nib.Nifti1Image:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["dcm2niix", "-z", "y", "-f", "%j", "-o", tmp, dicom_dir],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        produced = sorted(
            os.path.join(tmp, name)
            for name in os.listdir(tmp)
            if name.lower().endswith(NIFTI_SUFFIXES)
        )
        if not produced:
            raise RuntimeError(f"dcm2niix produced no NIfTI output for {dicom_dir}")
        if len(produced) > 1:
            # A split series (multi-echo, mixed orientations). Keep the volume
            # with the most slices, matching the notebook's fallback rule.
            def slice_count(candidate: str) -> int:
                shape = nib.load(candidate).shape
                return shape[2] if len(shape) > 2 else 0

            produced.sort(key=slice_count)
            logger.debug("dcm2niix split %s into %d volumes; keeping the thickest",
                         dicom_dir, len(produced))
        image = nib.load(produced[-1])
        # Force the data into memory before the temp directory disappears.
        return nib.Nifti1Image(np.asanyarray(image.dataobj), image.affine, image.header)


def _dicom_to_nifti_sitk(dicom_dir: str) -> nib.Nifti1Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        files = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not files:
            raise FileNotFoundError(f"No readable DICOM series in {dicom_dir}")
    else:
        if len(series_ids) > 1:
            logger.warning(
                "%d DICOM series found in %s; using the one with the most slices. "
                "Split the directory by SeriesInstanceUID to control this.",
                len(series_ids), dicom_dir,
            )
        files = max(
            (reader.GetGDCMSeriesFileNames(dicom_dir, uid) for uid in series_ids),
            key=len,
        )
    reader.SetFileNames(files)
    return sitk_to_nifti(reader.Execute())


def read_dicom_header(path: str):
    """Read the header of the first DICOM slice in `path` (file or directory)."""
    import pydicom

    if os.path.isdir(path):
        candidates = sorted(
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
            if not name.startswith(".") and name != "LICENSE"
        )
        if not candidates:
            raise FileNotFoundError(f"No files in {path}")
        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                return pydicom.dcmread(candidate, stop_before_pixels=True, force=False)
            except Exception as error:  # noqa: BLE001 - try the next file
                last_error = error
        raise ValueError(f"No readable DICOM file in {path}: {last_error}")
    return pydicom.dcmread(path, stop_before_pixels=True)


# ----------------------------------------------------------------------
# NIfTI <-> SimpleITK
# ----------------------------------------------------------------------
def nifti_to_sitk(image: nib.Nifti1Image) -> sitk.Image:
    """Convert to SimpleITK, translating the RAS affine into LPS geometry."""
    data = np.asanyarray(image.dataobj)
    ndim = data.ndim
    if ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D image, got {ndim}D with shape {data.shape}")

    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    lps = ras_to_lps @ image.affine

    rotation = lps[:ndim, :ndim]
    spacing = np.linalg.norm(rotation, axis=0)
    spacing[spacing == 0] = 1.0
    direction = rotation / spacing

    out = sitk.GetImageFromArray(np.ascontiguousarray(data.T))
    out.SetSpacing([float(value) for value in spacing])
    out.SetOrigin([float(value) for value in lps[:ndim, 3]])
    try:
        out.SetDirection([float(value) for value in direction.flatten()])
    except RuntimeError:
        # A 2D slice taken out of an oblique 3D volume can have a non-orthonormal
        # 2x2 block. Geometry is not meaningful for that case anyway.
        logger.debug("Non-orthonormal direction matrix; falling back to identity.")
    return out


def sitk_to_nifti(image: sitk.Image) -> nib.Nifti1Image:
    """Convert from SimpleITK, translating LPS geometry back into a RAS affine."""
    data = sitk.GetArrayFromImage(image).T
    ndim = image.GetDimension()

    spacing = np.asarray(image.GetSpacing(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(ndim, ndim)
    origin = np.asarray(image.GetOrigin(), dtype=float)

    lps = np.eye(4)
    lps[:ndim, :ndim] = direction * spacing
    lps[:ndim, 3] = origin

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return nib.Nifti1Image(data, lps_to_ras @ lps)


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
def save_image(
    image: nib.Nifti1Image,
    path: str,
    output_format: str = "nifti",
    compress: bool = True,
) -> str:
    """Write `image` to `path`, creating parent directories as needed.

    Returns the path actually written, which may differ from `path` if the
    extension had to be adjusted for `output_format`.
    """
    _check_output_format(output_format)
    path = resolve_output_path(path, output_format=output_format, compress=compress)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    if output_format in ("numpy", "npy"):
        np.save(path, np.asanyarray(image.dataobj))
    else:
        nib.save(image, path)
    return path


def resolve_output_path(path: str, output_format: str = "nifti", compress: bool = True) -> str:
    """Give `path` the extension implied by `output_format`."""
    _check_output_format(output_format)
    lower = path.lower()
    if output_format in ("numpy", "npy"):
        if lower.endswith(".npy"):
            return path
        return _strip_image_suffix(path) + ".npy"

    wanted = ".nii.gz" if compress else ".nii"
    if lower.endswith(wanted):
        return path
    return _strip_image_suffix(path) + wanted


def _check_output_format(output_format: str) -> None:
    """Reject a format we cannot write, rather than defaulting to NIfTI.

    Notably ``"dicom"``: DICOM is read-only here. A clipped, resampled,
    z-scored volume is no longer the acquisition the headers describe, so
    writing it back as DICOM would produce files that misrepresent
    themselves.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f"output_format must be one of {', '.join(OUTPUT_FORMATS)}, got "
            f"{output_format!r}."
            + (" DICOM output is not supported; ctkit reads DICOM but writes "
               "NIfTI or .npy." if "dicom" in output_format.lower() else "")
        )


def _strip_image_suffix(path: str) -> str:
    for suffix in (".nii.gz", ".nii", ".npy", ".npz", ".dcm", ".dicom", ".ima"):
        if path.lower().endswith(suffix):
            return path[: -len(suffix)]
    return path


def infer_series_id(source: Any) -> Optional[str]:
    """Guess a series identifier from a path.

    Files laid out as ``<series_id>/imaging.nii.gz`` take the directory name;
    otherwise the filename without image extensions is used.
    """
    if not isinstance(source, (str, os.PathLike)):
        return None
    path = os.path.abspath(str(source))
    if os.path.isdir(path):
        return os.path.basename(path.rstrip(os.sep))

    name = _strip_image_suffix(os.path.basename(path))
    parent = os.path.basename(os.path.dirname(path))
    generic = {"imaging", "image", "img", "ct", "volume", "data"}
    if name.lower() in generic and parent:
        return parent
    return name or parent or None
