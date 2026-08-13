"""The :class:`RadiologyImage` class: one scan, one optional mask, in memory.

Every processing step is a method that transforms the image (and its mask, when
the step affects geometry) and returns ``self``, so a protocol reads as a
chain::

    RadiologyImage("case/imaging.nii.gz", mask="case/segmentation.nii.gz") \\
        .orient().clip(-200, 300).resample((0.8, 0.8, 3.0)) \\
        .apply_mask().standardize_size(185, 185, 75) \\
        .save("out/case.nii.gz")

Nothing touches the disk until :meth:`RadiologyImage.save`.
"""

from __future__ import annotations

import copy as copy_module
import logging
import os
from typing import Any, Optional, Sequence, Union

import nibabel as nib
import numpy as np

from . import qc as qc_module
from . import segmentation as seg
from .config import ProcessingConfig
from .io import (
    ImageLike,
    infer_series_id,
    load_image,
    nifti_to_sitk,
    resolve_output_path,
    save_image,
    sitk_to_nifti,
)
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
    validate_class,
)

logger = logging.getLogger(__name__)


@validate_class
class RadiologyImage:
    """A single radiology image, optionally paired with a segmentation mask.

    Parameters
    ----------
    image:
        Path to a NIfTI file, a DICOM directory or zip, a ``.npy`` array, or an
        already-loaded ``Nifti1Image`` / ``SimpleITK.Image`` / ``ndarray``.
    mask:
        Optional segmentation in any of the same forms. By convention 1 is
        organ and 2 is tumor, but any labeling works as long as you pass the
        matching ``labels`` to the steps that use it.
    series_id:
        Identifier used in reports and output filenames. Inferred from the path
        when omitted.
    metadata:
        Free-form dictionary carried along with the image (a metadata row, the
        modality, acquisition parameters).
    masked_roi:
        Set when `mask` is not a label map but an image that keeps the original
        intensities inside the ROI and the image minimum outside it. It is
        converted to a real mask on load.
    """

    def __init__(
        self,
        image: ImageLike,
        mask: Optional[ImageLike] = None,
        series_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        masked_roi: bool = False,
        spacing: Optional[Sequence[float]] = None,
        affine: Optional[np.ndarray] = None,
        lazy: bool = False,
    ) -> None:
        self.source = image if isinstance(image, (str, os.PathLike)) else None
        self.mask_source = mask if isinstance(mask, (str, os.PathLike)) else None
        self.series_id = series_id or infer_series_id(image)
        self.metadata: dict = dict(metadata or {})
        self.history: list = []
        self._masked_roi = masked_roi
        self._spacing_hint = spacing
        self._affine_hint = affine

        if lazy and self.source is not None:
            self._image: Optional[nib.Nifti1Image] = None
            self._mask: Optional[nib.Nifti1Image] = None
            self._loaded = False
        else:
            self._image = load_image(image, affine=affine, spacing=spacing)
            self._mask = self._load_mask(mask)
            self._loaded = True

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _load_mask(self, mask: Optional[ImageLike]) -> Optional[nib.Nifti1Image]:
        if mask is None:
            return None
        loaded = load_image(mask, affine=self._affine_hint, spacing=self._spacing_hint)
        if self._masked_roi:
            loaded = seg.binarize_masked_image(loaded, label=seg.TUMOR)
        return loaded

    def load(self) -> "RadiologyImage":
        """Materialize a lazily-constructed image. Idempotent."""
        if not self._loaded:
            self._image = load_image(
                self.source, affine=self._affine_hint, spacing=self._spacing_hint
            )
            self._mask = self._load_mask(self.mask_source)
            self._loaded = True
        return self

    def unload(self) -> "RadiologyImage":
        """Drop pixel data, keeping identity and history. Only for path-backed images."""
        if self.source is None:
            raise ValueError(
                "Cannot unload an image that was not created from a path; its data "
                "would be lost. Save it first."
            )
        self._image = None
        self._mask = None
        self._loaded = False
        return self

    # ------------------------------------------------------------------
    # data access
    # ------------------------------------------------------------------
    @property
    def image(self) -> nib.Nifti1Image:
        """The image as a :class:`nibabel.Nifti1Image`."""
        self.load()
        assert self._image is not None
        return self._image

    @image.setter
    def image(self, value: ImageLike) -> None:
        self._image = load_image(value)
        self._loaded = True

    @property
    def mask(self) -> Optional[nib.Nifti1Image]:
        """The segmentation mask, or ``None``."""
        self.load()
        return self._mask

    @mask.setter
    def mask(self, value: Optional[ImageLike]) -> None:
        self.load()
        self._mask = None if value is None else load_image(value)

    @property
    def array(self) -> np.ndarray:
        """Voxel data as a NumPy array."""
        return np.asanyarray(self.image.dataobj)

    @property
    def mask_array(self) -> Optional[np.ndarray]:
        if self.mask is None:
            return None
        return np.asanyarray(self.mask.dataobj)

    @property
    def affine(self) -> np.ndarray:
        return self.image.affine

    @property
    def shape(self) -> tuple:
        return tuple(int(size) for size in self.image.shape)

    @property
    def ndim(self) -> int:
        return len(self.image.shape)

    @property
    def spacing(self) -> tuple:
        """Voxel size in mm along each axis."""
        return tuple(float(zoom) for zoom in self.image.header.get_zooms()[: self.ndim])

    @property
    def orientation(self) -> str:
        """Anatomical axis codes of the storage order, e.g. ``"RAS"``."""
        return "".join(nib.orientations.aff2axcodes(self.affine))

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def labels(self) -> list:
        """Sorted non-zero label values present in the mask."""
        if self.mask is None:
            return []
        values = np.unique(self.mask_array)
        return [int(value) for value in values if value != 0]

    # ------------------------------------------------------------------
    # conversions
    # ------------------------------------------------------------------
    def to_numpy(self, with_mask: bool = False):
        """Return the voxel array, or ``(image, mask)`` when `with_mask`."""
        if with_mask:
            return self.array, self.mask_array
        return self.array

    def to_nifti(self) -> nib.Nifti1Image:
        return self.image

    def to_sitk(self):
        return nifti_to_sitk(self.image)

    def copy(self) -> "RadiologyImage":
        """Deep copy, so chains can branch without interfering."""
        clone = copy_module.copy(self)
        clone.metadata = dict(self.metadata)
        clone.history = list(self.history)
        if self._loaded:
            clone._image = _copy_nifti(self._image)
            clone._mask = _copy_nifti(self._mask)
        return clone

    # ------------------------------------------------------------------
    # processing steps
    # ------------------------------------------------------------------
    def orient(self, target: str = "RAS") -> "RadiologyImage":
        """Reorient to a canonical anatomical axis order (default RAS).

        Archives are inconsistent about storage order, so two scans of the same
        anatomy can be mirrored or transposed relative to each other. This puts
        every image in the same frame, which any downstream model or feature
        will otherwise have to learn to undo.
        """
        target = target.upper()
        if self.orientation == target:
            return self._record("orient", target=target, changed=False)

        if target == "RAS":
            self._image = nib.as_closest_canonical(self.image)
            if self._mask is not None:
                self._mask = nib.as_closest_canonical(self._mask)
        else:
            self._image = _reorient_to(self.image, target)
            if self._mask is not None:
                self._mask = _reorient_to(self._mask, target)

        return self._record("orient", target=target, changed=True)

    def clip(
        self,
        min_value: Optional[Number] = -200,
        max_value: Optional[Number] = 300,
    ) -> "RadiologyImage":
        """Clamp intensities to a window, e.g. (-200, 300) HU for soft tissue.

        Clipping discards contrast the tissue of interest does not occupy, so
        the remaining range is spent where it matters. It also caps the effect
        of metal artifacts and out-of-field air.
        """
        if min_value is None and max_value is None:
            raise ValueError("clip() needs min_value and/or max_value")

        data = self.array.astype(_working_dtype(self.array.dtype), copy=True)
        np.clip(data, min_value, max_value, out=data)
        self._image = _with_data(self.image, data)
        return self._record("clip", min_value=min_value, max_value=max_value)

    def resample(
        self,
        spacing: Spacing = (0.8, 0.8, 3.0),
        interpolator: Interpolator = "linear",
    ) -> "RadiologyImage":
        """Resample onto a fixed voxel size in mm; ``None`` keeps an axis as is.

        CT series vary widely in slice thickness and in-plane resolution. Until
        they share a voxel grid, a millimeter of anatomy means a different
        number of voxels in each scan, and size or texture features are not
        comparable across a cohort.

        The mask, when present, is resampled with nearest-neighbor so its
        labels stay integral.
        """
        import SimpleITK as sitk

        spacing = list(spacing)
        if len(spacing) != self.ndim:
            spacing = (spacing + [None, None, None])[: self.ndim]

        current = self.spacing
        target = [
            float(current[axis]) if spacing[axis] is None else float(spacing[axis])
            for axis in range(self.ndim)
        ]
        if np.allclose(target, current, atol=1e-6):
            return self._record("resample", spacing=tuple(target), changed=False)

        interpolators = {
            "linear": sitk.sitkLinear,
            "nearest": sitk.sitkNearestNeighbor,
            "bspline": sitk.sitkBSpline,
        }

        self._image = _resample_nifti(self.image, target, interpolators[interpolator])
        if self._mask is not None:
            resampled_mask = _resample_nifti(self._mask, target, sitk.sitkNearestNeighbor)
            self._mask = _as_label_map(resampled_mask)

        return self._record("resample", spacing=tuple(target), changed=True)

    def segment(
        self,
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
    ) -> "RadiologyImage":
        """Segment organs with TotalSegmentator and set :attr:`mask`.

        Organs are labeled 1 and tumor 2. An existing mask (or `tumor_mask`)
        is treated as the tumor and merged in; with `restrict_organs_to_tumor`,
        only structures containing tumor are kept, so the healthy contralateral
        organ does not enter the ROI.

        Requires the ``TotalSegmentator`` package. Its files are written to a
        temporary directory and removed once the masks are in memory; pass
        `output_dir` to write them somewhere that is kept instead.
        """
        if organs is None:
            organs = _organs_from_metadata(self.metadata)
        if not organs:
            raise ValueError(
                "segment() needs `organs` (TotalSegmentator structure names). "
                "For a curated per-collection list use "
                "ProcessingConfig.for_dataset('tcga-kirc').organs"
            )

        existing_tumor = None
        if not replace:
            if tumor_mask is not None:
                existing_tumor = load_image(tumor_mask)
            elif self.mask is not None:
                existing_tumor = self.mask

        components = seg.segment_organ_components(
            self.image,
            organs,
            task=task,
            fast=fast,
            remove_small_blobs=remove_small_blobs,
            fill_holes=fill_holes,
            morphological_closing=morphological_closing,
            device=device,
            modality=str(self.metadata.get("Modality", self.metadata.get("modality", "CT"))),
            output_dir=output_dir,
        )
        organ_mask = seg.combine_organ_masks(list(components.values()))

        if existing_tumor is not None and existing_tumor.shape != organ_mask.shape:
            existing_tumor = seg.resample_mask_to(existing_tumor, self.image)

        self._mask = seg.combine_organ_and_tumor(
            organ_mask,
            existing_tumor,
            restrict_organs_to_tumor=restrict_organs_to_tumor,
            organ_components=components,
        )
        self.metadata["segmented_organs"] = list(components)
        return self._record(
            "segment", organs=list(components), task=task, has_tumor=existing_tumor is not None
        )

    def select_slice(
        self,
        mode: SliceMode = "mask",
        index: Optional[Integer] = None,
        label: Optional[Labels] = None,
        keepdims: bool = False,
    ) -> "RadiologyImage":
        """Reduce a volume to one axial slice — how a 3D series becomes a 2D example.

        Two ways to choose it:

        ``mode="mask"`` (the default)
            The slice holding the most mask. Needs a mask. `label` says which
            mask value counts; leave it out and the mask's only non-zero value
            is used, which covers a binary mask. A mask with several labels is
            ambiguous, so there `label` has to be given — ``2`` for the tumor
            in this package's organ=1 / tumor=2 convention, ``[1, 2]`` for
            both. If the requested label is absent, slice 0 is kept and the
            recorded count is zero, rather than failing.

        ``mode="index"``
            The slice you name in `index`. Negative indices count from the end.
            No mask needed.

        `keepdims` keeps the third axis with length 1 instead of dropping it.
        """
        if self.ndim < 3:
            return self._record("select_slice", mode=mode, changed=False)

        n_slices = self.shape[2]
        count: Optional[int] = None

        if mode == "index":
            if index is None:
                raise ValueError(
                    "select_slice(mode='index') needs index=<slice number>. Use "
                    "mode='mask' to pick the slice with the most mask instead."
                )
            chosen = int(index)
            if chosen < 0:
                chosen += n_slices
            if not 0 <= chosen < n_slices:
                raise IndexError(
                    f"{self.series_id or 'image'}: slice index {index} is out of range "
                    f"for a volume with {n_slices} slices."
                )
        else:
            if self.mask is None:
                raise ValueError(
                    "select_slice(mode='mask') needs a mask to measure. Call "
                    ".segment() first, pass mask= when constructing the image, or "
                    "use mode='index' to pick a slice number directly."
                )
            mask_data = self.mask_array
            label = self._resolve_mask_label(label)
            if label is None:  # an empty mask: nothing to measure
                chosen, count = 0, 0
            else:
                wanted = (
                    np.isin(mask_data, list(label))
                    if isinstance(label, (list, tuple, set, np.ndarray))
                    else (mask_data == label)
                )
                area_per_slice = wanted.sum(axis=(0, 1))
                chosen = int(np.argmax(area_per_slice))
                count = int(area_per_slice[chosen])
                if count == 0:
                    logger.warning(
                        "%s: label %s is absent from the mask; keeping slice 0.",
                        self.series_id or "image", _label_text(label),
                    )
                    chosen = 0

        self.metadata["selected_slice"] = chosen
        if mode == "mask":
            self.metadata["selected_slice_label"] = label
            self.metadata["selected_slice_mask_voxels"] = count

        selector = (slice(None), slice(None), slice(chosen, chosen + 1))
        image_slice = self.array[selector]
        if not keepdims:
            image_slice = image_slice[:, :, 0]

        affine = _shift_affine(self.affine, (0, 0, chosen))
        self._image = nib.Nifti1Image(image_slice, affine, self.image.header)
        if self._mask is not None:
            mask_slice = self.mask_array[selector]
            if not keepdims:
                mask_slice = mask_slice[:, :, 0]
            self._mask = nib.Nifti1Image(mask_slice, affine, self.mask.header)

        return self._record(
            "select_slice", mode=mode, index=chosen, label=label, n_voxels=count
        )

    def _resolve_mask_label(
        self, label: Optional[Union[int, Sequence[int]]]
    ) -> Optional[Union[int, Sequence[int]]]:
        """Default to the mask's only non-zero value; refuse to guess past that.

        Returns ``None`` when the mask is empty, so there is nothing to measure.
        """
        if label is not None:
            return label

        present = self.labels
        if len(present) == 1:
            return present[0]
        if not present:
            logger.warning(
                "%s: the mask is empty, so there is no label to select a slice by; "
                "keeping slice 0.",
                self.series_id or "image",
            )
            return None
        raise ValueError(
            f"{self.series_id or 'image'}: the mask holds several labels "
            f"({', '.join(str(value) for value in present)}), so which one to "
            "measure is ambiguous. Pass label= (in this package's convention 1 is "
            "organ and 2 is tumor; a list measures several together)."
        )

    def apply_mask(
        self,
        labels: Optional[Labels] = None,
        crop: bool = True,
        padding: Integer = 5,
        fill_value: Optional[Number] = None,
    ) -> "RadiologyImage":
        """Blank everything outside the mask, then crop to what is left.

        `labels` selects which mask values are kept (``None`` means every
        non-zero value). Voxels outside become `fill_value`, defaulting to the
        image minimum so the padding reads as air rather than as tissue.

        Cropping to the ROI bounding box (plus `padding` voxels) is what makes
        the output small enough to hold a cohort in memory.
        """
        if self.mask is None:
            raise ValueError(
                "apply_mask() needs a mask. Call .segment() first or pass mask= "
                "when constructing the image."
            )

        data = self.array
        mask_data = np.rint(self.mask_array).astype(np.int32)
        if data.shape != mask_data.shape:
            raise ValueError(
                f"Image shape {data.shape} does not match mask shape {mask_data.shape}. "
                "Resample them onto a common grid first."
            )

        if fill_value is None:
            fill_value = float(data.min())

        if labels is None:
            keep = mask_data > 0
        elif isinstance(labels, (list, tuple, set, np.ndarray)):
            keep = np.isin(mask_data, list(labels))
        else:
            keep = mask_data == labels

        if not keep.any():
            logger.warning(
                "%s: mask selects no voxels for labels=%s; the output will be empty.",
                self.series_id or "image", labels,
            )

        masked = np.where(keep, data, fill_value).astype(
            _working_dtype(data.dtype), copy=False
        )
        self._image = _with_data(self.image, masked)

        if crop:
            bbox = _bounding_box(masked, threshold=fill_value, padding=padding)
            if bbox is None:
                logger.warning(
                    "%s: nothing above the fill value after masking; skipping the crop.",
                    self.series_id or "image",
                )
            else:
                self._image = _crop(self._image, bbox)
                self._mask = _crop(self._mask, bbox)

        return self._record("apply_mask", labels=labels, crop=crop, padding=padding)

    def crop_to_content(
        self, threshold: Optional[Number] = None, padding: Integer = 5
    ) -> "RadiologyImage":
        """Crop to the bounding box of voxels above `threshold` (default: the minimum)."""
        data = self.array
        bbox = _bounding_box(data, threshold=threshold, padding=padding)
        if bbox is None:
            logger.warning("%s: image is uniform; skipping the crop.", self.series_id or "image")
            return self._record("crop_to_content", changed=False)
        self._image = _crop(self.image, bbox)
        if self._mask is not None:
            self._mask = _crop(self._mask, bbox)
        return self._record("crop_to_content", threshold=threshold, padding=padding)

    def standardize_size(
        self,
        x: Optional[Integer] = None,
        y: Optional[Integer] = None,
        z: Optional[Integer] = None,
        fill_value: Optional[Number] = None,
        mask_fill_value: Number = 0,
    ) -> "RadiologyImage":
        """Center-crop or center-pad to a fixed array shape.

        Models that take fixed-size tensors need every case to have the same
        shape. Cropping and padding about the center preserves voxel size (so
        the anatomy stays the same physical scale), unlike rescaling.
        ``None`` leaves an axis untouched.
        """
        dims = [x, y, z][: self.ndim]
        if all(dim is None for dim in dims):
            return self._record("standardize_size", changed=False)

        if fill_value is None:
            fill_value = float(self.array.min())

        self._image = _crop_or_pad(self.image, dims, fill_value)
        if self._mask is not None:
            self._mask = _crop_or_pad(self._mask, dims, mask_fill_value)
        return self._record("standardize_size", shape=self.shape)

    def normalize(
        self,
        method: NormalizationMethod = "volume",
        mean: Optional[Number] = None,
        std: Optional[Number] = None,
        within_mask: bool = False,
    ) -> "RadiologyImage":
        """Z-score the intensities.

        ``method="volume"`` uses this volume's own mean and standard deviation.
        ``method="dataset"`` requires `mean` and `std` computed over the whole
        cohort — use :meth:`Dataset.intensity_statistics`, which keeps every
        scan on one intensity scale so a model cannot key on per-scan offsets.
        """
        data = self.array.astype(np.float32, copy=True)

        if method == "dataset":
            if mean is None or std is None:
                raise ValueError(
                    "method='dataset' needs mean and std pooled over the cohort. "
                    "Get them from Dataset.intensity_statistics()."
                )
        else:
            sample = data[self.mask_array > 0] if (within_mask and self.mask is not None) else data
            mean = float(sample.mean())
            std = float(sample.std())

        if not std:
            raise ValueError(
                f"{self.series_id or 'image'}: standard deviation is zero, so the "
                "image is uniform and cannot be z-scored."
            )

        data -= float(mean)
        data /= float(std)
        self._image = _with_data(self.image, data)
        return self._record("normalize", method=method, mean=float(mean), std=float(std))

    # ------------------------------------------------------------------
    # whole pipeline
    # ------------------------------------------------------------------
    def process(
        self,
        config: Optional[Any] = None,
        **overrides: Any,
    ) -> "RadiologyImage":
        """Run the full protocol described by `config`, in order.

        `config` is a :class:`ProcessingConfig`, the name of a collection whose
        curated protocol to use (``image.process("tcga-kirc")``), or a path to
        a saved YAML protocol. Steps whose inputs are missing are skipped with
        a warning rather than raising, so one odd series does not stop a
        cohort. Pass keyword arguments to override individual config fields.
        """
        config = ProcessingConfig.resolve(config, **overrides)
        config.validate()

        if config.orient:
            self.orient(config.target_orientation)

        if config.segment:
            # One subdirectory per series, so that a cohort sharing a config
            # does not have every image overwrite the same filenames.
            segmentation_dir = (
                None
                if config.segmentation_dir is None
                else os.path.join(config.segmentation_dir, self.series_id or "image")
            )
            self.segment(
                organs=config.organs,
                task=config.totalsegmentator_task,
                restrict_organs_to_tumor=config.restrict_organs_to_tumor,
                fast=config.fast_segmentation,
                remove_small_blobs=config.remove_small_blobs,
                fill_holes=config.fill_holes,
                morphological_closing=config.morphological_closing,
                device=config.device,
                output_dir=segmentation_dir,
            )

        if config.clip:
            self.clip(config.clip_min, config.clip_max)

        if config.resample:
            self.resample(config.target_spacing)

        if config.dimensionality == "2D":
            if config.slice_selection_mode == "mask" and self.mask is None:
                raise ValueError(
                    f"{self.series_id or 'image'}: 2D output with "
                    "slice_selection_mode='mask' requires a mask to pick the slice "
                    "from. Enable segmentation, supply a mask, or set "
                    "slice_selection_mode='index' with slice_index=<number>."
                )
            self.select_slice(
                mode=config.slice_selection_mode,
                index=config.slice_index,
                label=config.slice_selection_label,
            )

        if config.mask:
            if self.mask is None:
                logger.warning(
                    "%s: mask=True but no mask is available; skipping the masking step.",
                    self.series_id or "image",
                )
            else:
                self.apply_mask(
                    labels=config.mask_labels,
                    crop=config.crop_to_mask,
                    padding=config.crop_padding,
                    fill_value=config.clip_min if config.clip else None,
                )

        if config.standardize_size:
            if config.target_shape is None:
                logger.warning(
                    "%s: standardize_size=True but target_shape is None. A single "
                    "image has no cohort to take percentiles over; skipping. Use "
                    "Dataset.process() or set config.target_shape.",
                    self.series_id or "image",
                )
            else:
                self.standardize_size(
                    *config.target_shape,
                    fill_value=config.clip_min if config.clip else None,
                )

        if config.normalize:
            self.normalize(
                method=config.normalization_method,
                mean=config.dataset_mean,
                std=config.dataset_std,
            )

        return self

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------
    def check(
        self,
        criteria: Union[qc_module.QCCriteria, QCPreset, None] = None,
        level: QCLevel = "all",
        **thresholds: Any,
    ) -> qc_module.QCResult:
        """Run quality control on this image. See :mod:`.qc`.

        `criteria` is a :class:`QCCriteria` or the name of a preset; single
        thresholds can be given as keywords, as in ``check(min_slices=25)``.

        `level` selects which checks run: ``"metadata"`` uses only the headers
        and :attr:`metadata`, so no pixel data is read; ``"volume"`` uses only
        the reconstructed volume; ``"all"`` (the default) runs both and merges
        the results.
        """
        criteria = qc_module.resolve_criteria(criteria, **thresholds)

        sources: list = []
        if level != "volume":
            if self.metadata:
                sources.append(self.metadata)
            if self.source is not None and os.path.isdir(str(self.source)):
                sources.append(str(self.source))

        return qc_module.check(
            None if level == "metadata" else self.image,
            criteria,
            series_id=self.series_id,
            metadata=sources or None,
        )

    def radiomics(
        self,
        labels: Labels = (1, 2),
        params: Optional[Union[str, dict]] = None,
        **kwargs: Any,
    ) -> dict:
        """Extract PyRadiomics features from the masked region. See :mod:`.features`."""
        from .features import extract_features

        return extract_features(self, labels=labels, params=params, **kwargs)

    def statistics(self) -> dict:
        """Summary statistics, useful for spotting inconsistent preprocessing."""
        data = self.array
        stats = {
            "series_id": self.series_id,
            "shape": self.shape,
            "spacing": self.spacing,
            "orientation": self.orientation,
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std()),
        }
        if self.mask is not None:
            mask_data = self.mask_array
            stats["mask_labels"] = self.labels
            stats["mask_voxels"] = int(np.count_nonzero(mask_data))
        return stats

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def save(
        self,
        path: PathArg,
        mask_path: Optional[PathArg] = None,
        output_format: OutputFormat = "nifti",
        compress: bool = True,
        save_mask: bool = True,
    ) -> str:
        """Write the processed image (and mask) — the only write in the pipeline.

        `path` may name a file or a directory; a directory gets
        ``<series_id>.nii.gz`` and ``<series_id>_mask.nii.gz``.

        Returns the image path written.
        """
        if os.path.isdir(path) or path.endswith(os.sep):
            name = self.series_id or "image"
            path = os.path.join(path, name)

        image_path = save_image(
            self.image, path, output_format=output_format, compress=compress
        )

        if save_mask and self.mask is not None:
            if mask_path is None:
                base = resolve_output_path(path, output_format=output_format, compress=compress)
                mask_path = _add_suffix(base, "_mask")
            save_image(self.mask, mask_path, output_format=output_format, compress=compress)

        return image_path

    def plot(
        self,
        z: Optional[int] = None,
        overlay_mask: bool = True,
        window: Optional[tuple] = None,
        out_path: Optional[str] = None,
        title: Optional[str] = None,
        ax=None,
    ):
        """Show one slice, optionally with the mask overlaid.

        With `z` omitted, the slice with the most mask is chosen (or the middle
        slice when there is no mask). Inside Jupyter, pass ``z=None`` and call
        :meth:`view` instead for an interactive slider.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        data = self.array
        mask_data = self.mask_array if (overlay_mask and self.mask is not None) else None

        if data.ndim == 2:
            plane, mask_plane, index = data, mask_data, 0
        else:
            if z is None:
                if mask_data is not None and np.any(mask_data):
                    z = int(np.argmax((mask_data > 0).sum(axis=(0, 1))))
                else:
                    z = data.shape[2] // 2
            index = int(z)
            plane = data[:, :, index]
            mask_plane = None if mask_data is None else mask_data[:, :, index]

        vmin, vmax = window if window else (None, None)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.rot90(plane), cmap="gray", vmin=vmin, vmax=vmax)
        if mask_plane is not None:
            overlay = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.7), (1, 0.5, 0, 0.9)])
            ax.imshow(
                np.rot90(np.clip(mask_plane, 0, 2)), cmap=overlay, alpha=0.35,
                vmin=0, vmax=2,
            )
        ax.axis("off")
        ax.set_title(title if title is not None else f"{self.series_id or 'image'} [z={index}]")

        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            ax.figure.savefig(out_path, bbox_inches="tight", dpi=200)
        return ax

    def view(self, overlay_mask: bool = True, window: Optional[tuple] = None):
        """Interactive slice slider for Jupyter notebooks."""
        from ipywidgets import interact

        if self.ndim < 3:
            return self.plot(overlay_mask=overlay_mask, window=window)

        def show(z: int):
            import matplotlib.pyplot as plt

            self.plot(z=z, overlay_mask=overlay_mask, window=window)
            plt.show()

        return interact(show, z=(0, self.shape[2] - 1))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _record(self, step: str, **details: Any) -> "RadiologyImage":
        self.history.append({"step": step, **details})
        return self

    @property
    def applied_steps(self) -> list:
        """Names of the steps applied so far, in order."""
        return [entry["step"] for entry in self.history]

    def filter(self, *args: Any, **kwargs: Any):
        """Not an image operation — defined only to say so.

        A method rather than ``__getattr__``, so that mypy and editors keep
        flagging every *other* misspelled attribute.
        """
        raise AttributeError(
            "filter() drops series from a cohort. For one image use check(), "
            "which returns a QCResult, or build a Dataset to filter over."
        )

    def __repr__(self) -> str:
        name = self.series_id or "unnamed"
        if not self._loaded:
            return f"<RadiologyImage {name} (not loaded)>"
        parts = [
            f"shape={self.shape}",
            f"spacing={tuple(round(value, 3) for value in self.spacing)}",
            f"orientation={self.orientation}",
        ]
        if self.mask is not None:
            parts.append(f"labels={self.labels}")
        if self.history:
            parts.append(f"steps={'>'.join(self.applied_steps)}")
        return f"<RadiologyImage {name} {' '.join(parts)}>"


# ----------------------------------------------------------------------
# array / geometry helpers
# ----------------------------------------------------------------------
def _label_text(label: Union[int, Sequence[int]]) -> str:
    if isinstance(label, (list, tuple, set, np.ndarray)):
        return ",".join(str(value) for value in label)
    return str(label)


def _working_dtype(dtype: np.dtype) -> np.dtype:
    """Keep floats as they are; promote integers to float32 for arithmetic."""
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float32)


def _with_data(reference: nib.Nifti1Image, data: np.ndarray) -> nib.Nifti1Image:
    """New image with `data`, keeping the geometry of `reference`."""
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    return nib.Nifti1Image(data, reference.affine, header)


def _copy_nifti(image: Optional[nib.Nifti1Image]) -> Optional[nib.Nifti1Image]:
    if image is None:
        return None
    return nib.Nifti1Image(
        np.array(np.asanyarray(image.dataobj), copy=True),
        image.affine.copy(),
        image.header.copy(),
    )


def _as_label_map(mask: nib.Nifti1Image) -> nib.Nifti1Image:
    data = np.rint(np.asanyarray(mask.dataobj)).astype(np.uint8)
    return _with_data(mask, data)


def _reorient_to(image: nib.Nifti1Image, target: str) -> nib.Nifti1Image:
    """Reorient to arbitrary axis codes such as ``"LPS"``."""
    try:
        target_ornt = nib.orientations.axcodes2ornt(tuple(target))
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Invalid orientation {target!r}; expected three axis codes such as "
            "'RAS', 'LPS' or 'LAS'."
        ) from error

    data = np.asanyarray(image.dataobj)
    current_ornt = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(image.affine))
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
    reoriented = nib.orientations.apply_orientation(data, transform)
    affine = image.affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
    return nib.Nifti1Image(reoriented, affine, image.header)


def _resample_nifti(
    image: nib.Nifti1Image, target_spacing: Sequence[float], interpolator
) -> nib.Nifti1Image:
    import SimpleITK as sitk

    source = nifti_to_sitk(image)
    original_spacing = source.GetSpacing()
    original_size = source.GetSize()

    new_size = [
        max(1, int(round(original_size[axis] * (original_spacing[axis] / target_spacing[axis]))))
        for axis in range(len(original_size))
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([float(value) for value in target_spacing])
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(source.GetDirection())
    resampler.SetOutputOrigin(source.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(source).min()))

    result = sitk_to_nifti(resampler.Execute(source))
    return nib.Nifti1Image(np.asanyarray(result.dataobj), result.affine, image.header)


def _bounding_box(
    data: np.ndarray, threshold: Optional[Number] = None, padding: int = 0
) -> Optional[tuple]:
    """Bounding box of voxels strictly above `threshold`, as (min, max) per axis."""
    if threshold is None:
        threshold = float(data.min())
    occupied = np.argwhere(data > threshold)
    if occupied.size == 0:
        return None

    mins = occupied.min(axis=0)
    maxs = occupied.max(axis=0) + 1
    if padding:
        mins = np.maximum(mins - padding, 0)
        maxs = np.minimum(maxs + padding, np.array(data.shape))
    return tuple((int(low), int(high)) for low, high in zip(mins, maxs))


def _crop(image: nib.Nifti1Image, bbox: Sequence[tuple]) -> nib.Nifti1Image:
    data = np.asanyarray(image.dataobj)
    selector = tuple(slice(low, high) for low, high in bbox)
    cropped = data[selector]
    offset = [low for low, _ in bbox]
    return nib.Nifti1Image(cropped, _shift_affine(image.affine, offset), image.header)


def _shift_affine(affine: np.ndarray, offset: Sequence[int]) -> np.ndarray:
    """Move the affine origin by a voxel `offset`, so world coordinates hold."""
    shifted = affine.copy()
    ndim = min(3, len(offset))
    voxel_offset = np.zeros(3)
    voxel_offset[:ndim] = offset[:ndim]
    shifted[:3, 3] = affine[:3, :3] @ voxel_offset + affine[:3, 3]
    return shifted


def _crop_or_pad(
    image: nib.Nifti1Image, dims: Sequence[Optional[int]], fill_value: Number
) -> nib.Nifti1Image:
    """Center-crop and/or center-pad each axis to `dims`."""
    data = np.asanyarray(image.dataobj)
    current = data.shape
    target = tuple(
        int(dims[axis]) if axis < len(dims) and dims[axis] is not None else current[axis]
        for axis in range(data.ndim)
    )

    out = np.full(target, fill_value, dtype=data.dtype)
    source_slices, dest_slices, offsets = [], [], []
    for axis in range(data.ndim):
        source_size, dest_size = current[axis], target[axis]
        if source_size >= dest_size:
            start = (source_size - dest_size) // 2
            source_slices.append(slice(start, start + dest_size))
            dest_slices.append(slice(0, dest_size))
            offsets.append(start)
        else:
            start = (dest_size - source_size) // 2
            source_slices.append(slice(0, source_size))
            dest_slices.append(slice(start, start + source_size))
            offsets.append(-start)

    out[tuple(dest_slices)] = data[tuple(source_slices)]
    return nib.Nifti1Image(out, _shift_affine(image.affine, offsets), image.header)


def _add_suffix(path: str, suffix: str) -> str:
    for extension in (".nii.gz", ".nii", ".npy"):
        if path.lower().endswith(extension):
            return path[: -len(extension)] + suffix + extension
    return path + suffix


def _organs_from_metadata(metadata: dict) -> Optional[list]:
    """Look up curated organ names when the image knows which collection it came from."""
    from .constants import tcia_dataset_to_info

    for key in ("dataset", "subproject", "collection", "Collection"):
        value = metadata.get(key)
        if not value:
            continue
        normalized = str(value).lower().replace("_", "-")
        info = tcia_dataset_to_info.get(normalized)
        if info and info.get("totalsegmentator_organs"):
            return list(info["totalsegmentator_organs"])
    return None
