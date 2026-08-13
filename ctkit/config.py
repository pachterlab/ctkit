"""Configuration objects for the processing pipeline.

A :class:`ProcessingConfig` is a plain, serializable description of *what* the
pipeline does. It carries no data and no state, so it can be written to YAML,
committed alongside a paper, and handed to a collaborator who wants to
reproduce a dataset exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Sequence, Union

from .constants import tcia_dataset_to_info

Number = Union[int, float]


@dataclass
class ProcessingConfig:
    """Every knob of the processing protocol, in the order it is applied.

    The defaults are the recommended protocol for abdominal CT: reorient to
    canonical RAS, clip to a soft-tissue window, resample to a common voxel
    grid, mask to the region of interest, standardize the array shape.

    Use :meth:`for_dataset` to get the values curated for a specific TCIA
    collection (organs to segment, clipping window, output dimensions).
    """

    # ---- 1. orientation ------------------------------------------------
    orient: bool = True
    target_orientation: str = "RAS"

    # ---- 2. segmentation ----------------------------------------------
    segment: bool = False
    organs: Optional[Sequence[str]] = None
    totalsegmentator_task: str = "total"
    remove_small_blobs: bool = True
    fill_holes: bool = True
    morphological_closing: bool = True
    restrict_organs_to_tumor: bool = True
    fast_segmentation: bool = False
    device: Optional[str] = None

    # ---- 3. intensity clipping ----------------------------------------
    clip: bool = True
    clip_min: Optional[Number] = -200
    clip_max: Optional[Number] = 300

    # ---- 4. resampling -------------------------------------------------
    resample: bool = True
    target_spacing: Sequence[Optional[Number]] = (0.8, 0.8, 3.0)

    # ---- 5. slice selection (2D only) ----------------------------------
    dimensionality: str = "3D"
    slice_selection_label: Union[int, Sequence[int]] = 2

    # ---- 6. masking ----------------------------------------------------
    mask: bool = True
    mask_labels: Optional[Union[int, Sequence[int]]] = None
    crop_to_mask: bool = True
    crop_padding: int = 5

    # ---- 7. size standardization ---------------------------------------
    standardize_size: bool = True
    target_shape: Optional[Sequence[Optional[int]]] = None
    shape_percentile: float = 95.0

    # ---- 8. intensity normalization -------------------------------------
    normalize: bool = False
    normalization_method: str = "volume"
    dataset_mean: Optional[float] = None
    dataset_std: Optional[float] = None

    # ---- output --------------------------------------------------------
    output_format: str = "nifti"
    compress: bool = True
    save_mask: bool = True
    dtype: str = "float32"

    # ---- radiomics -------------------------------------------------------
    radiomics_labels: Sequence[int] = (1, 2)
    radiomics_params: Optional[Union[str, dict]] = None

    # ---- provenance ------------------------------------------------------
    dataset: Optional[str] = None

    def __post_init__(self) -> None:
        self._normalize_sequences()
        self.validate()

    def _normalize_sequences(self) -> None:
        """Make sequence fields tuples regardless of how they were supplied.

        YAML has no tuple type, so without this a config would not compare
        equal to itself after a save/load round trip.
        """
        self.target_spacing = tuple(self.target_spacing)
        self.radiomics_labels = tuple(self.radiomics_labels)
        if self.target_shape is not None:
            self.target_shape = tuple(self.target_shape)
        if self.organs is not None:
            self.organs = list(self.organs)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self) -> "ProcessingConfig":
        """Raise :class:`ValueError` on internally inconsistent settings."""
        if self.dimensionality not in ("2D", "3D"):
            raise ValueError(
                f"dimensionality must be '2D' or '3D', got {self.dimensionality!r}"
            )
        if self.normalization_method not in ("volume", "dataset"):
            raise ValueError(
                "normalization_method must be 'volume' or 'dataset', got "
                f"{self.normalization_method!r}"
            )
        if self.output_format not in ("nifti", "numpy", "npy"):
            raise ValueError(
                f"output_format must be 'nifti' or 'numpy', got {self.output_format!r}"
            )
        if self.clip and self.clip_min is None and self.clip_max is None:
            raise ValueError(
                "clip=True requires clip_min and/or clip_max (e.g. -200/300 for a "
                "soft-tissue window). Set clip=False to skip clipping."
            )
        if self.segment and not self.organs:
            raise ValueError(
                "segment=True requires `organs` (TotalSegmentator structure names, "
                "e.g. ['kidney_left', 'kidney_right']). "
                "See ProcessingConfig.for_dataset() for curated per-collection values."
            )
        if len(self.target_spacing) != 3:
            raise ValueError(
                f"target_spacing must have 3 entries, got {self.target_spacing!r}"
            )
        if self.target_shape is not None and len(self.target_shape) != 3:
            raise ValueError(
                f"target_shape must have 3 entries (or be None), got {self.target_shape!r}"
            )
        return self

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def for_dataset(
        cls, dataset: str, masked: bool = True, **overrides: Any
    ) -> "ProcessingConfig":
        """Build a config from the curated settings for a TCIA collection.

        Parameters
        ----------
        dataset:
            Key of :data:`~ctkit.constants.tcia_dataset_to_info`,
            e.g. ``"tcga-kirc"``. Case- and separator-insensitive.
        masked:
            Whether the output will be masked to the ROI. Controls which set of
            curated output dimensions is used.
        **overrides:
            Any field of :class:`ProcessingConfig`.
        """
        key = _normalize_dataset_key(dataset)
        if key not in tcia_dataset_to_info:
            raise KeyError(
                f"Unknown dataset {dataset!r}. Known datasets: "
                f"{', '.join(sorted(tcia_dataset_to_info))}"
            )
        info = tcia_dataset_to_info[key]

        clip_min, clip_max = info.get("clip_min,clip_max", (None, None))
        shape_key = "xdim,ydim,zdim_masked" if masked else "xdim,ydim,zdim_unmasked"
        target_shape = info.get(shape_key)
        if target_shape is not None and all(d is None for d in target_shape):
            target_shape = None

        organs = list(info.get("totalsegmentator_organs") or [])

        params: dict = dict(
            clip=clip_min is not None or clip_max is not None,
            clip_min=clip_min,
            clip_max=clip_max,
            organs=organs or None,
            segment=bool(organs),
            totalsegmentator_task=info.get("totalsegmentator_task", "total"),
            mask=masked,
            target_shape=target_shape,
            dataset=key,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def radiomics(cls, dataset: Optional[str] = None, **overrides: Any) -> "ProcessingConfig":
        """Preset for radiomic feature extraction.

        PyRadiomics performs its own resampling, normalization and mask
        handling, so those steps are disabled here to avoid applying them
        twice. Orientation and clipping are kept.
        """
        params: dict = dict(
            resample=False,
            mask=False,
            standardize_size=False,
            normalize=False,
        )
        params.update(overrides)
        if dataset is not None:
            return cls.for_dataset(dataset, masked=False, **params)
        return cls(**params)

    @classmethod
    def minimal(cls, **overrides: Any) -> "ProcessingConfig":
        """Everything off: a no-op pipeline, useful as an ablation baseline."""
        params: dict = dict(
            orient=False,
            segment=False,
            clip=False,
            resample=False,
            mask=False,
            standardize_size=False,
            normalize=False,
        )
        params.update(overrides)
        return cls(**params)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProcessingConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown config keys: {', '.join(sorted(unknown))}. "
                f"Valid keys: {', '.join(sorted(known))}"
            )
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_yaml(self, path: Optional[str] = None) -> str:
        import yaml

        text = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if path:
            with open(path, "w") as handle:
                handle.write(text)
        return text

    @classmethod
    def from_yaml(cls, path: str) -> "ProcessingConfig":
        import yaml

        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    @classmethod
    def load(cls, path: str) -> "ProcessingConfig":
        """Load from a ``.yaml``/``.yml`` or ``.json`` file."""
        if path.endswith(".json"):
            with open(path) as handle:
                return cls.from_dict(json.load(handle))
        return cls.from_yaml(path)

    def replace(self, **overrides: Any) -> "ProcessingConfig":
        """Return a copy with `overrides` applied."""
        data = self.to_dict()
        data.update(overrides)
        return type(self).from_dict(data)

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    @property
    def steps(self) -> list:
        """Names of the steps that will actually run, in order."""
        enabled = []
        if self.orient:
            enabled.append("orient")
        if self.segment:
            enabled.append("segment")
        if self.clip:
            enabled.append("clip")
        if self.resample:
            enabled.append("resample")
        if self.dimensionality == "2D":
            enabled.append("select_slice")
        if self.mask:
            enabled.append("apply_mask")
        if self.standardize_size:
            enabled.append("standardize_size")
        if self.normalize:
            enabled.append("normalize")
        return enabled

    @property
    def needs_dataset_pass(self) -> bool:
        """True when a step needs statistics pooled over the whole dataset."""
        return bool(
            (self.normalize and self.normalization_method == "dataset"
             and self.dataset_mean is None)
            or (self.standardize_size and self.target_shape is None)
        )

    def describe(self) -> str:
        """Human-readable protocol summary, suitable for a methods section."""
        lines = ["Processing protocol:"]
        detail = {
            "orient": f"reorient to canonical {self.target_orientation}",
            "segment": (
                f"segment {', '.join(self.organs or [])} with TotalSegmentator"
                f" (task={self.totalsegmentator_task})"
            ),
            "clip": f"clip intensities to [{self.clip_min}, {self.clip_max}] HU",
            "resample": f"resample to {tuple(self.target_spacing)} mm voxels",
            "select_slice": (
                f"select the axial slice with the most label "
                f"{self.slice_selection_label}"
            ),
            "apply_mask": (
                "mask to "
                + ("all labels" if self.mask_labels is None else str(self.mask_labels))
                + (f", crop to ROI with {self.crop_padding} voxel padding"
                   if self.crop_to_mask else "")
            ),
            "standardize_size": (
                "crop/pad to "
                + (str(tuple(self.target_shape)) if self.target_shape
                   else f"the {self.shape_percentile:g}th percentile shape of the dataset")
            ),
            "normalize": f"z-score normalize per {self.normalization_method}",
        }
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"  {index}. {detail[step]}")
        if len(lines) == 1:
            lines.append("  (no steps enabled)")
        return "\n".join(lines)


def _normalize_dataset_key(dataset: str) -> str:
    """Map user spellings (``TCGA_KIRC``, ``TCGA-KIRC``) to registry keys."""
    key = str(dataset).strip().lower().replace("_", "-").replace(" ", "-")
    return key


# Backwards-compatible alias: the notebook parameter name.
DEFAULT_CONFIG = ProcessingConfig()
