"""The functional API: ``import ctkit`` and call the steps directly.

Every pipeline step is a function here, so a protocol can be written without
constructing anything::

    import ctkit

    scan = ctkit.orient("case/imaging.nii.gz")
    ctkit.clip(scan, -200, 300)
    ctkit.resample(scan, (0.8, 0.8, 3.0))
    ctkit.save(scan, "processed/case.nii.gz")

Each function takes whatever you have — a path, a NumPy array, a
:class:`~ctkit.image.RadiologyImage`, a :class:`~ctkit.dataset.Dataset`, or a
directory of scans — and returns the same kind of thing, so the same call works
on one scan or on a cohort::

    ctkit.orient(image)            # -> RadiologyImage
    ctkit.orient("data/raw")       # -> Dataset, every series reoriented

These are the methods, not a reimplementation of them: ``ctkit.clip(image)``
does exactly what ``image.clip()`` does, including modifying the image in
place and returning it. Construct a :class:`~ctkit.dataset.Dataset` or a
:class:`~ctkit.image.RadiologyImage` instead when you would rather chain::

    ctkit.Dataset("data/raw").filter(min_slices=25).orient().clip(-200, 300)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional, Sequence, Union

from .dataset import Dataset
from .image import RadiologyImage
from .io import ImageLike, is_dicom_directory
from .qc import QCCriteria, QCResult, resolve_criteria
from .qc import check as _check_metadata
from .validation import (
    Integer,
    Interpolator,
    Labels,
    NormalizationMethod,
    Number,
    OutputFormat,
    PathArg,
    QCLevel,
    QCPreset,
    SliceMode,
    Spacing,
    validated,
)

__all__ = [
    "filter",
    "check",
    "orient",
    "segment",
    "clip",
    "resample",
    "select_slice",
    "apply_mask",
    "crop_to_content",
    "standardize_size",
    "normalize",
    "process",
    "save",
    "radiomics",
]

#: Anything a function here accepts: one image, a cohort, or a path to either.
Target = Union[RadiologyImage, Dataset, ImageLike]


# ----------------------------------------------------------------------
# input handling
# ----------------------------------------------------------------------
def _resolve(target: Target) -> Union[RadiologyImage, Dataset]:
    """One image or a cohort, whichever the input describes.

    A path to a file is one image; a directory, a metadata table or a list is
    a cohort. A directory of DICOM slices is one image, not a cohort of one
    per file.
    """
    if isinstance(target, (RadiologyImage, Dataset)):
        return target
    if isinstance(target, (list, tuple)):
        return Dataset(target)
    if isinstance(target, (str, os.PathLike)):
        path = str(target)
        if os.path.isdir(path) and not is_dicom_directory(path):
            return Dataset(path)
        if path.lower().endswith((".csv", ".tsv")) or any(c in path for c in "*?["):
            return Dataset(path)
        return RadiologyImage(path)
    return RadiologyImage(target)


def _is_metadata(target: Any) -> bool:
    """True for a DICOM header or a metadata row, which carry no pixel data."""
    if isinstance(target, (RadiologyImage, Dataset, str, os.PathLike, list, tuple)):
        return False
    return isinstance(target, Mapping) or callable(getattr(target, "keys", None))


def _step(name: str, subject: Target, /, *args: Any, **kwargs: Any):
    """Run one step, on an image or over a cohort.

    Both classes carry the step under the same name, so this dispatches to the
    method rather than reimplementing it — which also means the arguments are
    validated here, before a cohort is even discovered.

    Positional-only, so a step's own arguments (``target=``, ``name=``) cannot
    collide with these.
    """
    return getattr(_resolve(subject), name)(*args, **kwargs)


# ----------------------------------------------------------------------
# quality control
# ----------------------------------------------------------------------
@validated
def filter(
    images: Target,
    criteria: Union[QCCriteria, QCPreset, None] = None,
    level: QCLevel = "all",
    progress: bool = True,
    **thresholds: Any,
) -> Dataset:
    """Drop series that fail quality control, returning the ones that passed.

    `criteria` is a :class:`QCCriteria`, or the name of a preset
    (``"default"``, ``"radiomics"``, ``"permissive"``). Individual thresholds
    can also be given as keywords: ``ctkit.filter(data, min_slices=25)``.

    `level` is ``"metadata"`` (headers only, reads no pixels), ``"volume"``, or
    ``"all"``. The result carries ``.qc_report``, the pass/fail table for every
    series, and ``.rejected``, the images that failed.
    """
    return Dataset(images).filter(
        criteria, level=level, progress=progress, **thresholds
    )


@validated
def check(
    target: Any,
    criteria: Union[QCCriteria, QCPreset, None] = None,
    level: QCLevel = "all",
    progress: bool = True,
    **thresholds: Any,
) -> Union[QCResult, Any]:
    """Run quality control without dropping anything.

    On one image this is a :class:`QCResult` (``.passed``, ``.reason``,
    ``.stats``); on a cohort it is the full pass/fail table as a DataFrame. A
    DICOM header or metadata row can be passed directly, which checks it
    without reading any pixels.
    """
    if _is_metadata(target):
        return _check_metadata(None, resolve_criteria(criteria, **thresholds), metadata=target)

    resolved = _resolve(target)
    if isinstance(resolved, Dataset):
        return resolved.check(criteria, level=level, progress=progress, **thresholds)
    return resolved.check(criteria, level=level, **thresholds)


# ----------------------------------------------------------------------
# pipeline steps
# ----------------------------------------------------------------------
@validated
def orient(image: Target, target: str = "RAS"):
    """Reorient to a canonical anatomical axis order. See :meth:`RadiologyImage.orient`."""
    return _step("orient", image, target=target)


@validated
def segment(
    image: Target,
    organs: Optional[Sequence[str]] = None,
    task: str = "total",
    tumor_mask: Optional[ImageLike] = None,
    restrict_organs_to_tumor: bool = True,
    fast: bool = False,
    remove_small_blobs: bool = True,
    fill_holes: bool = True,
    morphological_closing: bool = True,
    device: Optional[str] = None,
    output_dir: Optional[PathArg] = None,
    replace: bool = False,
):
    """Segment organs with TotalSegmentator. See :meth:`RadiologyImage.segment`.

    On a cohort, `output_dir` holds one subdirectory per series.
    """
    # Dataset.segment rather than _step: on a cohort it is the one that gives
    # each series its own output subdirectory.
    resolved = _resolve(image)
    return resolved.segment(
        organs=organs,
        task=task,
        tumor_mask=tumor_mask,
        restrict_organs_to_tumor=restrict_organs_to_tumor,
        fast=fast,
        remove_small_blobs=remove_small_blobs,
        fill_holes=fill_holes,
        morphological_closing=morphological_closing,
        device=device,
        output_dir=output_dir,
        replace=replace,
    )


@validated
def clip(
    image: Target,
    min_value: Optional[Number] = -200,
    max_value: Optional[Number] = 300,
):
    """Clamp intensities to a window in HU. See :meth:`RadiologyImage.clip`."""
    return _step("clip", image, min_value=min_value, max_value=max_value)


@validated
def resample(
    image: Target,
    spacing: Spacing = (0.8, 0.8, 3.0),
    interpolator: Interpolator = "linear",
):
    """Resample onto a fixed voxel size in mm. See :meth:`RadiologyImage.resample`."""
    return _step("resample", image, spacing=spacing, interpolator=interpolator)


@validated
def select_slice(
    image: Target,
    mode: SliceMode = "mask",
    index: Optional[Integer] = None,
    label: Optional[Labels] = None,
    keepdims: bool = False,
):
    """Reduce a volume to one axial slice. See :meth:`RadiologyImage.select_slice`."""
    return _step(
        "select_slice", image, mode=mode, index=index, label=label, keepdims=keepdims
    )


@validated
def apply_mask(
    image: Target,
    labels: Optional[Labels] = None,
    crop: bool = True,
    padding: Integer = 5,
    fill_value: Optional[Number] = None,
):
    """Blank outside the ROI and crop to it. See :meth:`RadiologyImage.apply_mask`."""
    return _step(
        "apply_mask", image, labels=labels, crop=crop, padding=padding, fill_value=fill_value
    )


@validated
def crop_to_content(image: Target, threshold: Optional[Number] = None, padding: Integer = 5):
    """Crop to the voxels above `threshold`. See :meth:`RadiologyImage.crop_to_content`."""
    return _step("crop_to_content", image, threshold=threshold, padding=padding)


@validated
def standardize_size(
    image: Target,
    x: Optional[Integer] = None,
    y: Optional[Integer] = None,
    z: Optional[Integer] = None,
    fill_value: Optional[Number] = None,
    mask_fill_value: Number = 0,
):
    """Center-crop or pad to a fixed shape. See :meth:`RadiologyImage.standardize_size`."""
    return _step(
        "standardize_size",
        image,
        x=x,
        y=y,
        z=z,
        fill_value=fill_value,
        mask_fill_value=mask_fill_value,
    )


@validated
def normalize(
    image: Target,
    method: NormalizationMethod = "volume",
    mean: Optional[Number] = None,
    std: Optional[Number] = None,
    within_mask: bool = False,
):
    """Z-score the intensities. See :meth:`RadiologyImage.normalize`.

    Over a cohort, ``method="dataset"`` needs pooled statistics, which
    :func:`process` resolves for you; here `mean` and `std` have to be given.
    """
    return _step(
        "normalize", image, method=method, mean=mean, std=std, within_mask=within_mask
    )


# ----------------------------------------------------------------------
# whole protocols, output, features
# ----------------------------------------------------------------------
@validated
def process(
    target: Target,
    config: Optional[Any] = None,
    **kwargs: Any,
):
    """Run a whole protocol, in the order the steps are listed.

    `config` is a :class:`ProcessingConfig`, the name of a collection whose
    curated protocol to use (``ctkit.process(data, "tcga-kirc")``), or a path
    to a saved YAML protocol. Any field can be overridden with a keyword.

    Over a cohort this is :meth:`Dataset.process`, so it also takes `out_dir`,
    `workers`, `skip_existing` and the rest, and it resolves the steps that
    need cohort-level statistics.
    """
    return _resolve(target).process(config, **kwargs)


@validated
def save(
    target: Target,
    path: PathArg,
    output_format: OutputFormat = "nifti",
    compress: bool = True,
    **kwargs: Any,
):
    """Write to `path` — a file for one image, a directory for a cohort.

    NIfTI (``.nii.gz``, or ``.nii`` with ``compress=False``) or a NumPy
    ``.npy`` array. Remaining keywords go to the method: `mask_path` and
    `save_mask` for one image, `layout` and `progress` for a cohort.
    """
    return _resolve(target).save(
        path, output_format=output_format, compress=compress, **kwargs
    )


@validated
def radiomics(
    target: Target,
    labels: Labels = (1, 2),
    params: Optional[Union[str, dict]] = None,
    **kwargs: Any,
):
    """Extract PyRadiomics features: a dict for one image, a DataFrame for a cohort."""
    return _resolve(target).radiomics(labels=labels, params=params, **kwargs)
