"""Argument validation for the public methods, via pydantic.

A processing step that is handed the wrong kind of argument should say so at
the call, naming the argument. Without this, a string where a number belongs
surfaces as a NumPy error several steps later, or — worse — as a silently
wrong volume.

Decorate a class with :func:`validate_class` — which applies pydantic's
``validate_call`` to its public methods — and annotate arguments with the
aliases here rather than bare ``int``/``float``/``str``: they accept what a
scientific caller actually has to hand (NumPy scalars, arrays, ``Path``
objects) and normalize it, so validation tightens the contract without
narrowing it.

The decorator returns the class unchanged as far as a type checker is
concerned, so editors and mypy still see every method with its real
signature.

Arguments with a fixed set of options — ``mode="mask"``, ``level="metadata"``,
an interpolator or output format — are ``Literal`` aliases, so the options are
part of the signature: pydantic rejects a bad one at the call and a type
checker rejects it before the code runs. Rules spanning several arguments
(``mode="index"`` needing an `index`, ``method="dataset"`` needing pooled
statistics) stay in the methods, where the message can say what to do.
"""

from __future__ import annotations

import functools
import os
import types
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
    get_args,
)

import numpy as np
from pydantic import BeforeValidator, ConfigDict, validate_call

__all__ = [
    "validate_class",
    "validated",
    # tolerant scalar types
    "Number",
    "Integer",
    "Labels",
    "Spacing",
    "PathArg",
    # closed sets of options
    "OutputFormat",
    "OUTPUT_FORMATS",
    "Layout",
    "Interpolator",
    "SliceMode",
    "NormalizationMethod",
    "Dimensionality",
    "QCLevel",
    "QCPreset",
    "OnError",
    "CacheMode",
]

#: Images, masks and metadata rows are third-party types (``Nifti1Image``,
#: ``SimpleITK.Image``, DICOM headers), so they are checked by isinstance
#: rather than parsed.
_CONFIG = ConfigDict(arbitrary_types_allowed=True)

_Class = TypeVar("_Class", bound=type)
_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def validated(func: _Callable) -> _Callable:
    """Validate one callable's arguments against its annotations.

    Typed as returning what it was given, so a type checker still sees the
    real signature of a decorated function.

    The pydantic validator is built on first call rather than at import, so
    that a method can be annotated with the class it is defined in — the usual
    ``-> "RadiologyImage"`` — and so that importing the package stays cheap.
    """
    validator: Optional[Callable] = None

    @functools.wraps(func)
    def call(*args: Any, **kwargs: Any):
        nonlocal validator
        if validator is None:
            validator = validate_call(config=_CONFIG)(func)
        return validator(*args, **kwargs)

    return call


def validate_class(cls: _Class) -> _Class:
    """Validate the arguments of every public method of `cls`.

    Constructors are left alone: they take deliberately polymorphic input (a
    path, an array, a directory, a table) and raise their own errors, which
    say more than a type mismatch would.

    Returns the same class, so a type checker sees the methods exactly as
    declared.
    """
    for name, attribute in list(vars(cls).items()):
        if name.startswith("_") or not isinstance(attribute, types.FunctionType):
            continue  # dunders, properties, classmethods and attributes
        setattr(cls, name, validated(attribute))
    return cls


def _to_scalar(value: Any) -> Any:
    """NumPy scalars become Python numbers; anything else is left alone.

    ``np.float32(0.8)`` and ``np.int64(185)`` are not instances of ``float``
    or ``int``, and reading a shape or a spacing out of an array is the most
    natural way to get one, so rejecting them would be a trap.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_sequence(value: Any) -> Any:
    """Arrays become lists, so a spacing or a shape can come from one."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _to_scalar(value)


def _to_path(value: Any) -> Any:
    """``Path`` objects become strings; the rest of the package uses os.path."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


#: An intensity, a spacing, a threshold — anything real-valued.
Number = Annotated[Union[int, float], BeforeValidator(_to_scalar)]
#: A count of voxels: a slice index, a padding, an axis length.
Integer = Annotated[int, BeforeValidator(_to_scalar)]
#: One mask value or several.
Labels = Annotated[Union[int, Sequence[int]], BeforeValidator(_to_sequence)]
#: A per-axis voxel size; ``None`` for an axis keeps that axis as it is.
Spacing = Annotated[Sequence[Optional[Number]], BeforeValidator(_to_sequence)]
#: A filesystem path, as a string or a ``Path``.
PathArg = Annotated[str, BeforeValidator(_to_path)]


# ----------------------------------------------------------------------
# closed sets of options
# ----------------------------------------------------------------------
# Spelling these out as types rather than checking them in the body means one
# declaration does both jobs: pydantic rejects a bad value at the call, and a
# type checker rejects it before the code runs.

#: How an image is written. ``"nii"`` is a synonym for ``"nifti"``; whether the
#: file is gzipped is `compress`.
OutputFormat = Literal["nifti", "nii", "numpy", "npy"]
#: The same options as a tuple, for the runtime checks in functions that are
#: not themselves validated.
OUTPUT_FORMATS = get_args(OutputFormat)
#: Where the files of a cohort go: ``out/<series_id>/imaging.nii.gz`` or
#: ``out/<series_id>.nii.gz``.
Layout = Literal["case_dirs", "flat"]
#: Interpolation for resampling. The mask always uses nearest-neighbor.
Interpolator = Literal["linear", "nearest", "bspline"]
#: How the 2D slice is chosen: the most mask, or the index you name.
SliceMode = Literal["mask", "index"]
#: Which statistics z-scoring uses.
NormalizationMethod = Literal["volume", "dataset"]
#: 2D slices or 3D volumes.
Dimensionality = Literal["2D", "3D"]
#: Which quality control checks run. ``"metadata"`` reads no pixel data.
QCLevel = Literal["metadata", "volume", "all"]
#: The named sets of quality control thresholds.
QCPreset = Literal["default", "radiomics", "permissive"]
#: What a cohort does with a series that fails.
OnError = Literal["warn", "raise"]
#: How protocols needing cohort-level statistics are staged.
CacheMode = Literal["auto", "disk", "none"]
