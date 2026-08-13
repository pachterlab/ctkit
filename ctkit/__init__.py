"""ctkit: reproducible CT image processing for AI and radiomics.

The package exists so that "how was this image processed?" has an exact,
runnable answer. A protocol is a :class:`ProcessingConfig`; applying it to one
scan is :class:`RadiologyImage`, and to a cohort is :class:`Dataset`.

Quick start — construct a cohort or a scan, then call the steps on it::

    import ctkit

    ctkit.download("tcga-kirc", "data/raw", limit=20)

    data = ctkit.Dataset("data/raw")
    data.filter(min_slices=25).process("tcga-kirc", out_dir="data/processed")

The same steps chain on one scan, modifying it in place::

    scan = ctkit.RadiologyImage("case/imaging.nii.gz", mask="case/segmentation.nii.gz")
    scan.orient().clip(-200, 300).resample((0.8, 0.8, 3.0)).save("processed/case.nii.gz")

Each step is also a function taking whatever you have — a path, an array, a
scan, or a cohort::

    ctkit.clip("case/imaging.nii.gz", -200, 300)

and a field of a :class:`ProcessingConfig`, which is what makes a protocol
something you can print, save to YAML, and publish alongside a paper.

Nothing is written to disk until ``save()``; intermediate volumes only ever
exist in memory.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

from .api import (
    apply_mask,
    check,
    clip,
    crop_to_content,
    filter,
    normalize,
    orient,
    process,
    radiomics,
    resample,
    save,
    segment,
    select_slice,
    standardize_size,
)
from .config import ProcessingConfig
from .constants import tcia_dataset_to_info
from .dataset import Dataset
from .datasets import EXTRA_CT_COLLECTIONS, get_dataset_info, list_datasets
from .image import RadiologyImage
from .io import dicom_to_nifti, load_image, save_image
from .metadata import annotate, categorize_phase, categorize_region, summarize
from .qc import QCCriteria, QCResult, check_series_metadata, check_volume

if TYPE_CHECKING:  # pragma: no cover
    from . import features, segmentation
    from .tcia import download, list_collections

__all__ = [
    "__version__",
    # pipeline steps, in the order they run
    "filter",
    "orient",
    "segment",
    "clip",
    "resample",
    "select_slice",
    "apply_mask",
    "crop_to_content",
    "standardize_size",
    "normalize",
    # whole protocols and output
    "process",
    "save",
    "radiomics",
    # core
    "RadiologyImage",
    "Dataset",
    "ProcessingConfig",
    # quality control
    "QCCriteria",
    "QCResult",
    "check",
    "check_volume",
    "check_series_metadata",
    # data access
    "download",
    "download_supplementary",
    "list_collections",
    "list_datasets",
    "get_dataset_info",
    "EXTRA_CT_COLLECTIONS",
    "tcia_dataset_to_info",
    # io
    "load_image",
    "save_image",
    "dicom_to_nifti",
    # metadata
    "categorize_region",
    "categorize_phase",
    "annotate",
    "summarize",
    # submodules
    "features",
    "segmentation",
    "tcia",
    "metadata",
]

#: Attributes served on first use, so that importing the package does not pull
#: in `requests`, `pyradiomics` or `matplotlib`.
_LAZY = {
    "download": ("tcia", "download"),
    "download_supplementary": ("tcia", "download_supplementary"),
    "list_collections": ("tcia", "list_collections"),
    "download_with_nbia_retriever": ("tcia", "download_with_nbia_retriever"),
    "get_series": ("tcia", "get_series"),
    "extract_features": ("features", "extract_features"),
    "segment_organs": ("segmentation", "segment_organs"),
}
_LAZY_MODULES = {"features", "segmentation", "tcia", "io", "qc", "datasets", "metadata"}


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        return getattr(import_module(f".{module_name}", __name__), attribute)
    if name in _LAZY_MODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(__all__) | set(globals()))
