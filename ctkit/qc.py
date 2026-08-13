"""Quality control: deciding which series are usable.

Two families of checks, matching the two points in a project where you can
cheaply throw data away:

* :func:`check_series_metadata` — needs only DICOM headers or a TCIA metadata
  row, so it can run *before* downloading or converting anything. Catches
  localizers, scouts, MIPs, sharp reconstruction kernels and thick slices.
* :func:`check_volume` — needs the reconstructed volume. Catches 4D series,
  too-few slices, extreme voxel spacing and anisotropic in-plane sampling.

Both return a :class:`QCResult`, which is falsy when the series should be
dropped and carries the reasons why. :func:`check` runs whichever of the two
applies to what you hand it and merges the outcomes;
:meth:`~ctkit.Dataset.filter` is the same thing over a cohort.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import replace as dataclass_replace
from typing import Any, Mapping, Optional, Sequence

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

#: Series whose description or image type contains any of these are never
#: usable for analysis: they are scanner-generated previews, not diagnostic
#: reconstructions.
BAD_SERIES_KEYWORDS = frozenset({
    "localizer", "survey", "asset", "scout", "cal", "mipseries", "pjn",
    "summary series", "topogram", "mip", "smart prep",
})

#: Sharp/bone reconstruction kernels. Fine for viewing bone, but their noise
#: texture dominates radiomic features, so they are excluded when texture
#: features matter.
SHARP_KERNEL_KEYWORDS = ("B5", "B6", "B7", "B8", "bone")


@dataclass
class QCResult:
    """Outcome of a quality check.

    Truthy when the series passed. ``reasons`` explains every failure, and
    ``stats`` carries the measurements taken along the way so they can be
    tabulated even for series that passed.
    """

    passed: bool = True
    reasons: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    series_id: Optional[str] = None

    def __bool__(self) -> bool:
        return self.passed

    def fail(self, reason: str) -> "QCResult":
        self.passed = False
        self.reasons.append(reason)
        return self

    @property
    def reason(self) -> str:
        """All failure reasons as one string (empty when passed)."""
        return "; ".join(self.reasons)

    def to_dict(self) -> dict:
        data = {"series_id": self.series_id, "passed": self.passed, "reason": self.reason}
        data.update(self.stats)
        return data

    def __repr__(self) -> str:
        verdict = "PASS" if self.passed else f"FAIL ({self.reason})"
        name = f"{self.series_id}: " if self.series_id else ""
        return f"<QCResult {name}{verdict}>"


@dataclass
class QCCriteria:
    """Thresholds for accepting a series.

    Defaults are the values used in the TCIA CT protocol. Set any field to
    ``None`` to disable that individual check.
    """

    # --- metadata-level ---
    modality: Optional[str] = "CT"
    min_slices: Optional[int] = 25
    max_slice_thickness: Optional[float] = 10.0
    exclude_keywords: Sequence[str] = tuple(sorted(BAD_SERIES_KEYWORDS))
    exclude_sharp_kernels: bool = False

    # --- volume-level ---
    reject_4d: bool = True
    max_spacing: Optional[float] = 20.0
    max_in_plane_anisotropy: Optional[float] = 4.0
    require_thickest_axis_is_superior_inferior: bool = False

    @classmethod
    def for_radiomics(cls, **overrides: Any) -> "QCCriteria":
        """Stricter criteria for texture analysis: drops sharp kernels too."""
        params: dict = dict(exclude_sharp_kernels=True)
        params.update(overrides)
        return cls(**params)

    @classmethod
    def permissive(cls, **overrides: Any) -> "QCCriteria":
        """Only drop series that cannot be processed at all."""
        params: dict = dict(
            modality=None,
            min_slices=1,
            max_slice_thickness=None,
            max_spacing=None,
            max_in_plane_anisotropy=None,
            exclude_keywords=(),
        )
        params.update(overrides)
        return cls(**params)


#: The presets, by the name you can pass instead of a `QCCriteria`.
PRESETS = {
    "default": QCCriteria,
    "radiomics": QCCriteria.for_radiomics,
    "permissive": QCCriteria.permissive,
}


def resolve_criteria(
    criteria: Optional[Any] = None, **thresholds: Any
) -> QCCriteria:
    """A :class:`QCCriteria` from an object, a preset name, or bare thresholds.

    This is what lets ``filter(min_slices=25)`` and ``filter("radiomics")``
    mean the same thing as building the object yourself.
    """
    if isinstance(criteria, str):
        if criteria not in PRESETS:
            raise ValueError(
                f"Unknown quality control preset {criteria!r}; "
                f"expected one of {', '.join(sorted(PRESETS))}."
            )
        base = PRESETS[criteria]()
    else:
        base = criteria or QCCriteria()

    if not thresholds:
        return base
    known = {item.name for item in dataclass_fields(QCCriteria)}
    unknown = sorted(set(thresholds) - known)
    if unknown:
        raise TypeError(
            f"Unknown quality control threshold(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(known))}."
        )
    return dataclass_replace(base, **thresholds)


# ----------------------------------------------------------------------
# metadata-level checks
# ----------------------------------------------------------------------
def check_series_metadata(
    metadata: Any,
    criteria: Optional[QCCriteria] = None,
    series_id: Optional[str] = None,
) -> QCResult:
    """Check a series from headers alone — no pixel data required.

    `metadata` may be a :class:`pydicom.Dataset`, a mapping (a row of TCIA
    ``getSeries`` output or of a metadata CSV), or a path to a DICOM file or
    directory.
    """
    criteria = criteria or QCCriteria()
    result = QCResult(series_id=series_id)

    if isinstance(metadata, (str, os.PathLike)):
        return _check_dicom_path(str(metadata), criteria, result)

    fields = _extract_metadata_fields(metadata)
    result.stats.update({k: v for k, v in fields.items() if v is not None})
    _apply_metadata_criteria(fields, criteria, result)
    return result


def _check_dicom_path(path: str, criteria: QCCriteria, result: QCResult) -> QCResult:
    from .io import read_dicom_header

    if os.path.isdir(path):
        slices = [
            name for name in os.listdir(path)
            if not name.startswith(".") and name != "LICENSE"
        ]
        if criteria.min_slices is not None and len(slices) < criteria.min_slices:
            result.fail(f"too few DICOM files ({len(slices)} < {criteria.min_slices})")
        result.stats["n_files"] = len(slices)
        if not slices:
            return result

    try:
        header = read_dicom_header(path)
    except Exception as error:  # noqa: BLE001 - unreadable is a QC failure
        return result.fail(f"unreadable DICOM: {error}")

    fields = _extract_metadata_fields(header)
    # A directory listing counts slices more reliably than the header does.
    if "n_files" in result.stats:
        fields["image_count"] = result.stats["n_files"]
    result.stats.update({k: v for k, v in fields.items() if v is not None})
    _apply_metadata_criteria(fields, criteria, result, count_checked="n_files" in result.stats)
    return result


def _extract_metadata_fields(metadata: Any) -> dict:
    """Pull the fields we filter on out of a DICOM header or a metadata row."""

    def lookup(*names: str) -> Any:
        for name in names:
            if isinstance(metadata, Mapping):
                if name in metadata and _is_present(metadata[name]):
                    return metadata[name]
            else:
                value = getattr(metadata, name, None)
                if _is_present(value):
                    return value
        return None

    text_parts = []
    for value in (
        lookup("SeriesDescription", "Series Description", "series_description"),
        lookup("ProtocolName", "Protocol Name", "protocol_name"),
        lookup("StudyDescription", "Study Description", "study_description"),
        lookup("ImageType", "Image Type", "image_type"),
    ):
        if value is None:
            continue
        if isinstance(value, str):
            text_parts.extend(value.split("\\"))
        elif isinstance(value, (list, tuple)):
            text_parts.extend(str(item) for item in value)
        else:
            text_parts.append(str(value))

    thickness = lookup("SliceThickness", "Slice Thickness", "slice_thickness")
    count = lookup(
        "ImageCount", "Number of Images", "Number of Images Original",
        "image_count", "NumberOfFrames",
    )

    return {
        "modality": _as_str(lookup("Modality", "modality")),
        "series_description": _as_str(
            lookup("SeriesDescription", "Series Description", "series_description")
        ),
        "slice_thickness": _as_float(thickness),
        "image_count": _as_int(count),
        "_text": [part.lower() for part in text_parts if part],
    }


def _apply_metadata_criteria(
    fields: dict,
    criteria: QCCriteria,
    result: QCResult,
    count_checked: bool = False,
) -> None:
    text = fields.get("_text") or []
    result.stats.pop("_text", None)

    if criteria.modality is not None and fields.get("modality"):
        if str(fields["modality"]).upper() != criteria.modality.upper():
            result.fail(f"modality is {fields['modality']}, not {criteria.modality}")

    for keyword in criteria.exclude_keywords or ():
        if any(_mentions(part, keyword) for part in text):
            result.fail(f"excluded keyword {keyword!r} in series description/image type")
            break

    if criteria.exclude_sharp_kernels:
        for keyword in SHARP_KERNEL_KEYWORDS:
            if any(keyword.lower() in part for part in text):
                result.fail(f"sharp reconstruction kernel {keyword!r}")
                break

    thickness = fields.get("slice_thickness")
    if criteria.max_slice_thickness is not None and thickness is not None:
        if thickness > criteria.max_slice_thickness:
            result.fail(
                f"slice thickness {thickness:g} mm > {criteria.max_slice_thickness:g} mm"
            )

    count = fields.get("image_count")
    if not count_checked and criteria.min_slices is not None and count is not None:
        if count < criteria.min_slices:
            result.fail(f"too few slices ({count} < {criteria.min_slices})")


# ----------------------------------------------------------------------
# volume-level checks
# ----------------------------------------------------------------------
def check_volume(
    image: Any,
    criteria: Optional[QCCriteria] = None,
    series_id: Optional[str] = None,
) -> QCResult:
    """Check a reconstructed volume: dimensionality, slice count and spacing."""
    criteria = criteria or QCCriteria()
    result = QCResult(series_id=series_id)

    if image is None:
        return result.fail("image is missing")

    if not isinstance(image, nib.Nifti1Image):
        from .io import load_image

        try:
            image = load_image(image)
        except Exception as error:  # noqa: BLE001 - unloadable is a QC failure
            return result.fail(f"could not load image: {error}")

    try:
        shape = image.shape
        zooms = tuple(float(z) for z in image.header.get_zooms()[: len(shape)])
    except Exception as error:  # noqa: BLE001 - corrupt header is a QC failure
        return result.fail(f"could not read image header: {error}")

    result.stats["shape"] = tuple(int(s) for s in shape)
    result.stats["spacing"] = zooms
    orientation = nib.orientations.aff2axcodes(image.affine)
    result.stats["orientation"] = "".join(str(code) for code in orientation)

    if len(shape) > 3:
        result.stats["is_4d"] = True
        if criteria.reject_4d:
            result.fail(
                f"4D volume with shape {tuple(shape)} — the DICOM series probably "
                "mixes phases or time points and should be split first"
            )
            return result
    else:
        result.stats["is_4d"] = False

    n_slices = int(shape[2]) if len(shape) > 2 else 1
    result.stats["n_slices"] = n_slices
    if criteria.min_slices is not None and n_slices < criteria.min_slices:
        result.fail(f"only {n_slices} slices (< {criteria.min_slices})")

    if zooms:
        max_spacing = max(zooms)
        result.stats["max_spacing"] = max_spacing
        if criteria.max_spacing is not None and max_spacing > criteria.max_spacing:
            result.fail(f"voxel spacing {max_spacing:g} mm > {criteria.max_spacing:g} mm")

    if len(shape) >= 3:
        canonical_zooms = np.asarray(
            nib.as_closest_canonical(image).header.get_zooms()[:3], dtype=float
        )
        in_plane = canonical_zooms[:2]
        anisotropy = (
            float(in_plane.max() / in_plane.min()) if in_plane.min() > 0 else np.inf
        )
        result.stats["in_plane_anisotropy"] = anisotropy
        if (criteria.max_in_plane_anisotropy is not None
                and anisotropy > criteria.max_in_plane_anisotropy):
            result.fail(
                f"in-plane anisotropy {anisotropy:.2f} > "
                f"{criteria.max_in_plane_anisotropy:g} — likely a reformat or an "
                "off-axis acquisition"
            )

        thickest_axis = int(np.argmax(zooms))
        thickest_code = str(orientation[thickest_axis]) if thickest_axis < len(orientation) else ""
        in_si = thickest_code in ("S", "I")
        result.stats["thickest_axis_is_si"] = in_si
        if criteria.require_thickest_axis_is_superior_inferior and not in_si:
            result.fail(
                f"thickest axis is {thickest_code or 'unknown'}, not superior-inferior "
                "(the series is probably coronal or sagittal)"
            )

    return result


def check(
    image: Any = None,
    criteria: Optional[QCCriteria] = None,
    series_id: Optional[str] = None,
    metadata: Any = None,
) -> QCResult:
    """Run every applicable check and merge the outcomes into one result.

    `image` is a volume, a path to one, or a DICOM directory; `metadata` is a
    DICOM header, a metadata row, a path, or a list of those. A DICOM directory
    passed as `image` is used as a metadata source as well, since its headers
    and file count are the cheapest checks available.

    Pass ``image=None`` to check metadata alone — no pixel data is read, which
    is what makes it usable before a collection has been downloaded. With
    nothing to check the result passes: quality control drops series that are
    demonstrably unusable, not series nothing is known about.
    """
    criteria = criteria or QCCriteria()

    sources = list(metadata) if isinstance(metadata, (list, tuple)) else [metadata]
    if isinstance(image, (str, os.PathLike)) and os.path.isdir(str(image)):
        sources.append(str(image))

    results = [
        check_series_metadata(source, criteria, series_id=series_id)
        for source in sources
        if source is not None and not (isinstance(source, Mapping) and not source)
    ]
    if image is not None:
        results.append(check_volume(image, criteria, series_id=series_id))

    if not results:
        logger.debug("%s: nothing to check.", series_id or "image")
        return QCResult(series_id=series_id)
    return merge(results, series_id=series_id)


def merge(results: Sequence[QCResult], series_id: Optional[str] = None) -> QCResult:
    """Combine several results: passes only if all of them pass."""
    merged = QCResult(series_id=series_id)
    for result in results:
        merged.passed = merged.passed and result.passed
        merged.reasons.extend(result.reasons)
        merged.stats.update(result.stats)
        if merged.series_id is None:
            merged.series_id = result.series_id
    return merged


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _mentions(text: str, keyword: str) -> bool:
    """Whole-word keyword match.

    Series descriptions are terse and abbreviated, so several of the keywords
    are short (``cal``, ``mip``, ``pjn``). Matched as bare substrings they hit
    innocent words — ``cal`` is inside ``cervical`` and ``apical`` — and drop
    usable series. Word boundaries keep the abbreviations usable.
    """
    return re.search(rf"\b{re.escape(keyword.lower())}\b", text) is not None


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if isinstance(value, float) and np.isnan(value):
            return False
    except TypeError:
        pass
    return not (isinstance(value, str) and not value.strip())


def _as_str(value: Any) -> Optional[str]:
    if not _is_present(value):
        return None
    return str(value)


def _as_float(value: Any) -> Optional[float]:
    if not _is_present(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    number = _as_float(value)
    return None if number is None else int(number)
