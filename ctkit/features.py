"""Radiomic feature extraction with PyRadiomics.

PyRadiomics accepts SimpleITK images directly, so features are computed from
the in-memory volume without writing the processed image to disk first.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .image import RadiologyImage

logger = logging.getLogger(__name__)

#: PyRadiomics settings used in the protocol. ``resampledPixelSpacing`` is set
#: per call from the image dimensionality.
DEFAULT_RADIOMICS_PARAMS = {
    "imageType": {"Original": {}},
    "setting": {
        "binWidth": 25,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": "sitkBSpline",
        "normalize": False,
        "padDistance": 5,
    },
}


def default_params(dimensionality: str = "3D") -> dict:
    """Protocol defaults, with the resampling grid matched to `dimensionality`."""
    params = {
        "imageType": {key: dict(value) for key, value in DEFAULT_RADIOMICS_PARAMS["imageType"].items()},
        "setting": dict(DEFAULT_RADIOMICS_PARAMS["setting"]),
    }
    if dimensionality == "2D":
        params["setting"]["resampledPixelSpacing"] = [1, 1]
        params["setting"]["force2D"] = True
    return params


def extract_features(
    image: Union["RadiologyImage", Any],
    labels: Sequence[int] = (1, 2),
    params: Optional[Union[str, dict]] = None,
    mask: Optional[Any] = None,
    drop_diagnostics: bool = False,
) -> dict:
    """Extract features from one image/mask pair.

    Parameters
    ----------
    image:
        A :class:`~ctkit.image.RadiologyImage`, or anything
        :func:`~ctkit.io.load_image` accepts.
    labels:
        Mask values that make up the region of interest. Several values are
        merged into a single region before extraction, so ``(1, 2)`` measures
        organ and tumor together; pass ``2`` for tumor only.
    params:
        Path to a PyRadiomics parameter YAML, or a settings dict. Defaults to
        :func:`default_params`.
    drop_diagnostics:
        Remove the ``diagnostics_*`` provenance entries from the result.

    Returns
    -------
    dict
        Feature name to value, plus ``series_id`` when the input carries one.
    """
    from radiomics import featureextractor

    from .image import RadiologyImage
    from .io import nifti_to_sitk

    if isinstance(image, RadiologyImage):
        radiology_image = image
    else:
        radiology_image = RadiologyImage(image, mask=mask)

    if radiology_image.mask is None:
        raise ValueError(
            f"{radiology_image.series_id or 'image'}: radiomics needs a mask defining "
            "the region of interest."
        )

    dimensionality = "2D" if radiology_image.ndim < 3 else "3D"
    sitk_image = nifti_to_sitk(radiology_image.image)
    sitk_mask, label = _merge_labels(radiology_image, labels)

    extractor = _build_extractor(
        featureextractor, params if params is not None else default_params(dimensionality)
    )

    with _quiet_radiomics():
        features = extractor.execute(sitk_image, sitk_mask, label=label)

    result = {
        key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
        for key, value in features.items()
        if not (drop_diagnostics and key.startswith("diagnostics_"))
    }
    if radiology_image.series_id:
        result = {"series_id": radiology_image.series_id, **result}
    return result


def _merge_labels(radiology_image: "RadiologyImage", labels: Sequence[int]):
    """Collapse the requested labels into a single region valued 1."""
    from .io import nifti_to_sitk

    if isinstance(labels, (int, np.integer)):
        wanted = [int(labels)]
    else:
        wanted = [int(value) for value in labels]

    mask_data = np.rint(radiology_image.mask_array).astype(np.int32)
    present = set(int(value) for value in np.unique(mask_data)) - {0}
    missing = [value for value in wanted if value not in present]
    if missing and len(missing) == len(wanted):
        raise ValueError(
            f"{radiology_image.series_id or 'image'}: none of the labels {wanted} are "
            f"in the mask (it contains {sorted(present) or 'nothing'})."
        )
    if missing:
        logger.debug(
            "%s: labels %s absent from the mask; extracting from %s.",
            radiology_image.series_id or "image", missing,
            [value for value in wanted if value in present],
        )

    merged = np.isin(mask_data, wanted).astype(np.uint8)

    import nibabel as nib

    mask_image = nib.Nifti1Image(merged, radiology_image.mask.affine, radiology_image.mask.header)
    return nifti_to_sitk(mask_image), 1


def _build_extractor(featureextractor, params: Union[str, dict]):
    if isinstance(params, str):
        if not os.path.exists(params):
            raise FileNotFoundError(f"PyRadiomics parameter file not found: {params}")
        return featureextractor.RadiomicsFeatureExtractor(params)

    import yaml

    # PyRadiomics only reads a nested imageType/featureClass/setting structure
    # from a file, so a dict is written to a short-lived temporary file.
    with tempfile.TemporaryDirectory(prefix="ctkit_radiomics_") as tmp:
        path = os.path.join(tmp, "params.yaml")
        with open(path, "w") as handle:
            yaml.safe_dump(params, handle, sort_keys=False, default_flow_style=False)
        return featureextractor.RadiomicsFeatureExtractor(path)


class _quiet_radiomics:
    """Silence PyRadiomics' very chatty per-image logging."""

    def __enter__(self):
        import radiomics

        self._level = radiomics.logger.level
        radiomics.logger.setLevel(logging.ERROR)
        return self

    def __exit__(self, *exc_info) -> None:
        import radiomics

        radiomics.logger.setLevel(self._level)
