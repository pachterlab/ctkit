"""The :class:`Dataset` class: a cohort of images processed with one protocol.

A dataset is lazy by default. It holds paths, not pixels, and loads one image
at a time, so a cohort that would never fit in memory can still be processed
in a single call::

    Dataset.from_directory("data/nifti") \\
        .filter() \\
        .process(config, out_dir="data/processed")

Two of the steps in the protocol need statistics pooled over the whole cohort
— the output shape (a percentile of the observed shapes) and dataset-level
z-scoring. :meth:`Dataset.process` resolves those automatically, making extra
passes over the data when it has to.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Union

import numpy as np

from .config import ProcessingConfig
from .image import RadiologyImage
from .io import NIFTI_SUFFIXES
from .qc import QCCriteria, QCResult, check_volume

logger = logging.getLogger(__name__)

#: Filenames treated as "the image" when scanning a directory of case folders.
DEFAULT_IMAGE_NAMES = (
    "imaging.nii.gz", "imaging.nii", "image.nii.gz", "image.nii",
    "ct.nii.gz", "ct.nii", "volume.nii.gz",
)
#: Filenames treated as "the mask" for a case folder.
DEFAULT_MASK_NAMES = (
    "segmentation.nii.gz", "segmentation.nii", "mask.nii.gz", "mask.nii",
    "label.nii.gz", "labels.nii.gz", "seg.nii.gz",
)
MASK_SUFFIXES = ("_mask", "_seg", "_segmentation", "_label", "_labels")


class Dataset:
    """An ordered collection of :class:`RadiologyImage` objects.

    Parameters
    ----------
    images:
        ``RadiologyImage`` objects, or anything the constructor accepts
        (paths, arrays, NIfTI images).
    name:
        Optional label used in log messages.
    lazy:
        Keep images unloaded until they are used. Leave this on for cohorts
        that do not fit in memory.
    """

    def __init__(
        self,
        images: Iterable[Any],
        name: Optional[str] = None,
        lazy: bool = True,
    ) -> None:
        self.name = name
        self.lazy = lazy
        self.images: list = [self._coerce(item, lazy) for item in images]
        self.qc_report = None

    @staticmethod
    def _coerce(item: Any, lazy: bool) -> RadiologyImage:
        if isinstance(item, RadiologyImage):
            return item
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return RadiologyImage(item[0], mask=item[1], lazy=lazy)
        if isinstance(item, dict):
            return RadiologyImage(**item)
        return RadiologyImage(item, lazy=lazy)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        root: str,
        image_pattern: Optional[str] = None,
        mask_pattern: Optional[str] = None,
        recursive: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "Dataset":
        """Discover images under `root`.

        Three layouts are recognized automatically:

        * one directory per case holding ``imaging.nii.gz`` and (optionally)
          ``segmentation.nii.gz`` — the layout this package writes;
        * a flat directory of NIfTI files, where a file whose name ends in
          ``_mask``/``_seg``/``_label`` is paired with its image;
        * one directory per case holding DICOM slices.

        `image_pattern` and `mask_pattern` are glob patterns (e.g.
        ``"*_CT.nii.gz"``) that override the automatic rules.
        """
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Not a directory: {root}")

        if image_pattern or mask_pattern:
            pairs = _scan_with_patterns(root, image_pattern, mask_pattern, recursive)
        else:
            pairs = _scan_case_directories(root)
            if not pairs:
                pairs = _scan_flat_files(root)
            if not pairs:
                pairs = _scan_dicom_directories(root)

        if not pairs:
            raise FileNotFoundError(
                f"No images found under {root}. Expected per-case directories "
                f"containing one of {DEFAULT_IMAGE_NAMES}, a flat directory of NIfTI "
                "files, or per-case DICOM directories. Pass image_pattern= to "
                "override."
            )

        images = [
            RadiologyImage(image, mask=mask, series_id=series_id, lazy=True, **kwargs)
            for series_id, image, mask in pairs
        ]
        return cls(images, name=name or os.path.basename(os.path.abspath(root)))

    @classmethod
    def from_paths(
        cls,
        images: Sequence[Any],
        masks: Optional[Sequence[Any]] = None,
        series_ids: Optional[Sequence[str]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "Dataset":
        """Build from explicit lists of images and (optionally) masks."""
        if masks is not None and len(masks) != len(images):
            raise ValueError(
                f"Got {len(images)} images but {len(masks)} masks; they must match."
            )
        if series_ids is not None and len(series_ids) != len(images):
            raise ValueError(
                f"Got {len(images)} images but {len(series_ids)} series_ids."
            )

        built = [
            RadiologyImage(
                image,
                mask=None if masks is None else masks[index],
                series_id=None if series_ids is None else series_ids[index],
                lazy=True,
                **kwargs,
            )
            for index, image in enumerate(images)
        ]
        return cls(built, name=name)

    @classmethod
    def from_metadata(
        cls,
        metadata,
        root: Optional[str] = None,
        image_column: str = "Image",
        mask_column: str = "Mask",
        series_id_column: str = "series_id",
        name: Optional[str] = None,
    ) -> "Dataset":
        """Build from a metadata table (a DataFrame or a path to a CSV).

        Every column becomes part of each image's :attr:`RadiologyImage.metadata`,
        so acquisition parameters travel with the scan through the pipeline.
        """
        import pandas as pd

        frame = pd.read_csv(metadata) if isinstance(metadata, str) else metadata
        if image_column not in frame.columns:
            raise KeyError(
                f"Column {image_column!r} not in the metadata "
                f"(columns: {', '.join(map(str, frame.columns))})"
            )

        def resolve(value: Any) -> Optional[str]:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            path = str(value)
            if root and not os.path.isabs(path):
                path = os.path.join(root, path)
            return path

        images = []
        for _, row in frame.iterrows():
            image_path = resolve(row[image_column])
            if image_path is None:
                continue
            mask_path = resolve(row[mask_column]) if mask_column in frame.columns else None
            images.append(
                RadiologyImage(
                    image_path,
                    mask=mask_path,
                    series_id=str(row[series_id_column]) if series_id_column in frame.columns else None,
                    metadata=row.to_dict(),
                    lazy=True,
                )
            )
        return cls(images, name=name)

    # ------------------------------------------------------------------
    # sequence protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[RadiologyImage]:
        return iter(self.images)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Dataset(self.images[index], name=self.name, lazy=self.lazy)
        return self.images[index]

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        with_masks = sum(1 for image in self.images if image.mask_source or image._mask is not None)
        return f"<Dataset{label} n={len(self)} with_masks={with_masks}>"

    @property
    def series_ids(self) -> list:
        return [image.series_id for image in self.images]

    def get(self, series_id: str) -> RadiologyImage:
        """Look up one image by series id."""
        for image in self.images:
            if image.series_id == series_id:
                return image
        raise KeyError(f"No series {series_id!r} in this dataset")

    # ------------------------------------------------------------------
    # quality control
    # ------------------------------------------------------------------
    def filter(
        self,
        criteria: Optional[QCCriteria] = None,
        progress: bool = True,
        keep_report: bool = True,
    ) -> "Dataset":
        """Drop series that fail quality control, returning a new dataset.

        The full pass/fail table, including the measurements taken for series
        that passed, is left on :attr:`qc_report` of the returned dataset (and
        of this one), so exclusions can be reported in a paper.
        """
        criteria = criteria or QCCriteria()
        records, kept = [], []

        for image in _progress(self.images, "Quality control", progress):
            try:
                result = check_volume(image.image, criteria, series_id=image.series_id)
            except Exception as error:  # noqa: BLE001 - unreadable is a failure
                result = QCResult(series_id=image.series_id).fail(f"failed to load: {error}")
            records.append(result.to_dict())
            if result.passed:
                kept.append(image)
            else:
                logger.info("Excluding %s: %s", image.series_id, result.reason)
            if self.lazy and image.source is not None:
                image.unload()

        report = _to_frame(records)
        filtered = Dataset(kept, name=self.name, lazy=self.lazy)
        if keep_report:
            self.qc_report = report
            filtered.qc_report = report

        logger.info(
            "Quality control kept %d of %d series (%d excluded).",
            len(kept), len(self.images), len(self.images) - len(kept),
        )
        return filtered

    # ------------------------------------------------------------------
    # cohort statistics
    # ------------------------------------------------------------------
    def shape_percentile(self, percentile: float = 95.0, progress: bool = True) -> tuple:
        """Per-axis percentile of the image shapes across the cohort.

        This is how a common output size is chosen: large enough to contain
        almost every ROI, small enough that the rare huge case is cropped
        rather than every case being padded to it.
        """
        shapes = []
        for image in _progress(self.images, f"Measuring shapes (p{percentile:g})", progress):
            data = image.array
            if data.size and np.all(data == data.flat[0]):
                logger.warning("%s is uniform; excluding it from the shape statistics.",
                               image.series_id)
            else:
                shapes.append(image.shape)
            if self.lazy and image.source is not None:
                image.unload()

        if not shapes:
            raise ValueError("No usable images to measure shapes from")

        n_axes = max(len(shape) for shape in shapes)
        padded = np.array([list(shape) + [1] * (n_axes - len(shape)) for shape in shapes])
        return tuple(int(np.percentile(padded[:, axis], percentile)) for axis in range(n_axes))

    def intensity_statistics(self, progress: bool = True) -> dict:
        """Mean and standard deviation pooled over every voxel in the cohort.

        Computed in one streaming pass with running sums, so it does not need
        the cohort in memory.
        """
        total = 0
        total_sum = 0.0
        total_square_sum = 0.0

        for image in _progress(self.images, "Pooling intensities", progress):
            data = np.asarray(image.array, dtype=np.float64)
            total += data.size
            total_sum += float(data.sum())
            total_square_sum += float(np.square(data).sum())
            if self.lazy and image.source is not None:
                image.unload()

        if total == 0:
            raise ValueError("No voxels to compute statistics from")

        mean = total_sum / total
        variance = max(total_square_sum / total - mean * mean, 0.0)
        return {"mean": mean, "std": float(np.sqrt(variance)), "n_voxels": total}

    def statistics(self, progress: bool = True):
        """Per-image summary statistics as a DataFrame."""
        records = []
        for image in _progress(self.images, "Summarizing", progress):
            records.append(image.statistics())
            if self.lazy and image.source is not None:
                image.unload()
        return _to_frame(records)

    def check_consistency(
        self,
        mean_tolerance: float = 100.0,
        std_fraction_tolerance: float = 0.15,
        progress: bool = True,
    ) -> list:
        """Flag images whose intensity distribution is unlike the rest.

        A series that survives every other check can still be wrong — a
        different contrast phase, a failed rescale, an inverted intensity
        scale. Comparing each volume against the cohort median catches those.
        """
        frame = self.statistics(progress=progress)
        if frame is None or len(frame) == 0:
            return []

        median_mean = float(np.median(frame["mean"]))
        median_std = float(np.median(frame["std"]))
        problems = []
        for _, row in frame.iterrows():
            if abs(row["mean"] - median_mean) > mean_tolerance:
                problems.append(
                    f"{row['series_id']}: mean {row['mean']:.1f} differs from the cohort "
                    f"median {median_mean:.1f}"
                )
            if abs(row["std"] - median_std) > std_fraction_tolerance * median_std:
                problems.append(
                    f"{row['series_id']}: std {row['std']:.1f} differs from the cohort "
                    f"median {median_std:.1f}"
                )

        if problems:
            logger.warning("Intensity consistency check found %d issues:", len(problems))
            for problem in problems:
                logger.warning("  %s", problem)
        else:
            logger.info("Intensity consistency check passed for all %d series.", len(frame))
        return problems

    # ------------------------------------------------------------------
    # processing
    # ------------------------------------------------------------------
    def resolve_config(
        self, config: ProcessingConfig, progress: bool = True
    ) -> ProcessingConfig:
        """Fill in the config fields that depend on the whole cohort.

        Resolves ``target_shape`` (from the shape percentile) and
        ``dataset_mean``/``dataset_std``, each measured at the point in the
        pipeline where the corresponding step runs.
        """
        resolved = config

        if config.standardize_size and config.target_shape is None:
            logger.info(
                "target_shape is unset: measuring the p%g shape across %d series.",
                config.shape_percentile, len(self),
            )
            probe = resolved.replace(standardize_size=False, normalize=False)
            shapes = self._processed_view(probe, progress=progress).shape_percentile(
                config.shape_percentile, progress=progress
            )
            shapes = (list(shapes) + [None, None, None])[:3]
            resolved = resolved.replace(target_shape=tuple(shapes))
            logger.info("Resolved target_shape=%s", tuple(shapes))

        if (config.normalize and config.normalization_method == "dataset"
                and config.dataset_mean is None):
            logger.info("Pooling cohort intensity statistics for dataset normalization.")
            probe = resolved.replace(normalize=False)
            stats = self._processed_view(probe, progress=progress).intensity_statistics(
                progress=progress
            )
            resolved = resolved.replace(
                dataset_mean=stats["mean"], dataset_std=stats["std"]
            )
            logger.info(
                "Resolved dataset_mean=%.4f dataset_std=%.4f", stats["mean"], stats["std"]
            )

        return resolved

    def _processed_view(self, config: ProcessingConfig, progress: bool = True) -> "Dataset":
        """A dataset of images processed with `config`, materialized on demand.

        Used for the measurement passes. The processed volumes are produced one
        at a time and released, never written to disk.
        """
        return _ProcessedView(self, config, progress=progress)

    @staticmethod
    def _split_config(config: ProcessingConfig):
        """Split a protocol into the part before the cohort-level steps and the rest.

        The expensive steps — segmentation above all — live in the first half.
        Running them once, caching the result, and applying only the second half
        afterwards keeps a cohort-level protocol to a single segmentation pass.
        """
        head = config.replace(standardize_size=False, normalize=False)
        tail = config.replace(
            orient=False,
            segment=False,
            clip=False,
            resample=False,
            mask=False,
            dimensionality="3D",  # the slice, if any, was already selected
        )
        return head, tail

    def process(
        self,
        config: Optional[ProcessingConfig] = None,
        out_dir: Optional[str] = None,
        workers: int = 1,
        progress: bool = True,
        layout: str = "case_dirs",
        skip_existing: bool = False,
        on_error: str = "warn",
        cache: str = "auto",
        cache_dir: Optional[str] = None,
        **overrides: Any,
    ) -> "Dataset":
        """Run the protocol over the cohort.

        Parameters
        ----------
        config:
            The protocol. Defaults to :class:`ProcessingConfig` defaults.
        out_dir:
            Where to write the processed images. Only the final image (and its
            mask) is written — no intermediate files. When omitted, results
            are kept in memory and returned, which needs the whole cohort to
            fit in RAM.
        workers:
            Processes to run in parallel. Leave at 1 when segmenting on a
            single GPU.
        layout:
            ``"case_dirs"`` writes ``out_dir/<series_id>/imaging.nii.gz``;
            ``"flat"`` writes ``out_dir/<series_id>.nii.gz``.
        skip_existing:
            Skip series whose output already exists, so an interrupted run can
            be resumed.
        on_error:
            ``"warn"`` logs and continues to the next series, ``"raise"`` stops.
        cache:
            How to handle protocols with cohort-level steps, which need more
            than one look at the data. ``"auto"`` (the default) stages the
            expensive part of the pipeline in a temporary directory when the
            protocol segments, so TotalSegmentator runs once instead of three
            times; ``"disk"`` always stages; ``"none"`` recomputes instead,
            trading time for temporary disk. The staging directory is deleted
            before this returns either way.
        cache_dir:
            Where to stage. Defaults to the system temporary directory.

        Returns
        -------
        Dataset
            The processed cohort. Backed by the written files when `out_dir`
            was given.
        """
        config = (config or ProcessingConfig())
        if overrides:
            config = config.replace(**overrides)
        config.validate()

        if on_error not in ("warn", "raise"):
            raise ValueError(f"on_error must be 'warn' or 'raise', got {on_error!r}")
        if layout not in ("case_dirs", "flat"):
            raise ValueError(f"layout must be 'case_dirs' or 'flat', got {layout!r}")
        if cache not in ("auto", "disk", "none"):
            raise ValueError(f"cache must be 'auto', 'disk' or 'none', got {cache!r}")

        if config.needs_dataset_pass:
            logger.info(
                "This protocol has cohort-level steps (%s), so the data is read more "
                "than once. Set target_shape/dataset_mean/dataset_std in the config to "
                "process in a single pass.",
                ", ".join(
                    step for step in ("standardize_size", "normalize")
                    if step in config.steps
                ),
            )
            staging = cache == "disk" or (cache == "auto" and config.segment)
            if staging:
                return self._process_staged(
                    config, out_dir, workers, progress, layout, skip_existing,
                    on_error, cache_dir,
                )
            config = self.resolve_config(config, progress=progress)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            config.to_yaml(os.path.join(out_dir, "processing_config.yaml"))

        if workers > 1 and out_dir:
            results = self._process_parallel(
                config, out_dir, workers, progress, layout, skip_existing, on_error
            )
        else:
            if workers > 1:
                logger.warning(
                    "workers>1 needs out_dir (results are passed back as files); "
                    "processing sequentially instead."
                )
            results = self._process_sequential(
                config, out_dir, progress, layout, skip_existing, on_error
            )

        processed = Dataset(
            [item for item in results if item is not None],
            name=self.name,
            lazy=bool(out_dir),
        )
        if out_dir:
            self._write_manifest(processed, out_dir)
        return processed

    def _process_staged(
        self, config, out_dir, workers, progress, layout, skip_existing,
        on_error, cache_dir,
    ) -> "Dataset":
        """Run a cohort-level protocol with one pass over the expensive steps.

        The pipeline is split in two. The first half — orientation,
        segmentation, clipping, resampling, masking — runs once into a
        temporary directory. The cohort statistics are then measured from that
        staged output, and only the cheap second half (crop/pad and z-scoring)
        is applied to produce the final files. The staging directory is removed
        before returning.
        """
        import shutil
        import tempfile

        head, tail = self._split_config(config)
        staging = tempfile.mkdtemp(prefix="ctkit_stage_", dir=cache_dir)
        logger.info(
            "Staging the per-image steps (%s) once in a temporary directory, so they "
            "are not repeated for each cohort-level measurement.",
            ", ".join(head.steps) or "none",
        )
        try:
            staged = self.process(
                head,
                out_dir=staging,
                workers=workers,
                progress=progress,
                layout="case_dirs",
                on_error=on_error,
                cache="none",
            )
            # Measure on the staged images. `tail` has the per-image steps
            # switched off, so the probe passes over the staged data are
            # no-ops and nothing expensive is repeated.
            resolved_tail = staged.resolve_config(tail, progress=progress)
            resolved = config.replace(
                target_shape=resolved_tail.target_shape,
                dataset_mean=resolved_tail.dataset_mean,
                dataset_std=resolved_tail.dataset_std,
            )
            processed = staged.process(
                resolved_tail,
                out_dir=out_dir,
                workers=workers,
                progress=progress,
                layout=layout,
                skip_existing=skip_existing,
                on_error=on_error,
                cache="none",
            )
            if out_dir:
                # Record the protocol as a whole, not just the half applied last.
                resolved.to_yaml(os.path.join(out_dir, "processing_config.yaml"))
            return processed
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _process_sequential(
        self, config, out_dir, progress, layout, skip_existing, on_error
    ) -> list:
        results = []
        for image in _progress(self.images, "Processing", progress):
            try:
                results.append(
                    _process_one_image(image, config, out_dir, layout, skip_existing)
                )
            except Exception as error:  # noqa: BLE001 - one bad series must not stop a cohort
                if on_error == "raise":
                    raise
                logger.error("Failed to process %s: %s", image.series_id, error)
                results.append(None)
            finally:
                if self.lazy and image.source is not None and out_dir:
                    image.unload()
        return results

    def _process_parallel(
        self, config, out_dir, workers, progress, layout, skip_existing, on_error
    ) -> list:
        unpicklable = [image for image in self.images if image.source is None]
        if unpicklable:
            logger.warning(
                "%d images hold in-memory data and cannot be sent to worker "
                "processes; processing sequentially instead.", len(unpicklable),
            )
            return self._process_sequential(
                config, out_dir, progress, layout, skip_existing, on_error
            )

        payloads = [
            {
                "source": str(image.source),
                "mask_source": None if image.mask_source is None else str(image.mask_source),
                "series_id": image.series_id,
                "metadata": image.metadata,
                "masked_roi": image._masked_roi,
                "config": config.to_dict(),
                "out_dir": out_dir,
                "layout": layout,
                "skip_existing": skip_existing,
            }
            for image in self.images
        ]

        results: list = [None] * len(payloads)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_payload, payload): index
                for index, payload in enumerate(payloads)
            }
            for future in _progress(
                as_completed(futures), "Processing", progress, total=len(futures)
            ):
                index = futures[future]
                try:
                    outcome = future.result()
                except Exception as error:  # noqa: BLE001
                    if on_error == "raise":
                        raise
                    logger.error(
                        "Failed to process %s: %s", payloads[index]["series_id"], error
                    )
                    continue
                results[index] = RadiologyImage(
                    outcome["image_path"],
                    mask=outcome.get("mask_path"),
                    series_id=outcome["series_id"],
                    metadata=outcome.get("metadata") or {},
                    lazy=True,
                )
        return results

    def _write_manifest(self, processed: "Dataset", out_dir: str) -> None:
        records = []
        for image in processed:
            record = {"series_id": image.series_id, "image": image.source}
            if image.mask_source:
                record["mask"] = image.mask_source
            record.update(
                {key: value for key, value in image.metadata.items()
                 if not isinstance(value, (list, dict, np.ndarray))}
            )
            records.append(record)
        frame = _to_frame(records)
        if frame is not None:
            frame.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------
    def radiomics(
        self,
        labels: Sequence[int] = (1, 2),
        params: Optional[Union[str, dict]] = None,
        out_csv: Optional[str] = None,
        progress: bool = True,
        on_error: str = "warn",
    ):
        """Extract radiomic features for every series, as a DataFrame."""
        from .features import extract_features

        records = []
        for image in _progress(self.images, "Extracting features", progress):
            try:
                features = extract_features(image, labels=labels, params=params)
            except Exception as error:  # noqa: BLE001
                if on_error == "raise":
                    raise
                logger.error("Feature extraction failed for %s: %s", image.series_id, error)
                continue
            finally:
                if self.lazy and image.source is not None:
                    image.unload()
            records.append(features)

        frame = _to_frame(records)
        if out_csv and frame is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
            frame.to_csv(out_csv, index=False)
            logger.info("Wrote %d feature rows to %s", len(frame), out_csv)
        return frame

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def map(
        self, function: Callable[[RadiologyImage], Any], progress: bool = True
    ) -> list:
        """Apply `function` to each image, returning the results."""
        outputs = []
        for image in _progress(self.images, "Mapping", progress):
            outputs.append(function(image))
            if self.lazy and image.source is not None:
                image.unload()
        return outputs

    def save(
        self,
        out_dir: str,
        layout: str = "case_dirs",
        output_format: str = "nifti",
        compress: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        """Write every image in the dataset to `out_dir`."""
        os.makedirs(out_dir, exist_ok=True)
        written = []
        for image in _progress(self.images, "Saving", progress):
            path = _output_path(image, out_dir, layout, output_format, compress)
            image.save(path, output_format=output_format, compress=compress)
            written.append(image)
        return Dataset(written, name=self.name, lazy=self.lazy)

    @property
    def metadata(self):
        """Per-image metadata as a DataFrame."""
        return _to_frame(
            [{"series_id": image.series_id, **image.metadata} for image in self.images]
        )


class _ProcessedView(Dataset):
    """A dataset whose images are processed with a config as they are read.

    Used for the measurement passes in :meth:`Dataset.resolve_config`. Nothing
    is cached: each access reprocesses, which trades compute for memory and
    keeps the promise that no intermediate files are written.
    """

    def __init__(self, source: Dataset, config: ProcessingConfig, progress: bool = True):
        self._source = source
        self._config = config
        self.name = source.name
        self.lazy = True
        self.qc_report = None
        self.images = [_ProcessedProxy(image, config) for image in source.images]


class _ProcessedProxy(RadiologyImage):
    """A :class:`RadiologyImage` that runs a config the first time it loads."""

    def __init__(self, source_image: RadiologyImage, config: ProcessingConfig):
        self._source_image = source_image
        self._config = config
        super().__init__(
            source_image.source if source_image.source is not None else np.zeros((1, 1, 1)),
            mask=source_image.mask_source,
            series_id=source_image.series_id,
            metadata=dict(source_image.metadata),
            masked_roi=source_image._masked_roi,
            lazy=True,
        )
        # An in-memory source has no path to lazily read back from, so mark it
        # unloaded here and copy the pixels in load().
        self._image = None
        self._mask = None
        self._loaded = False

    def load(self) -> "RadiologyImage":
        if not self._loaded:
            if self._source_image.source is not None:
                super().load()
            else:
                original = self._source_image.copy()
                self._image = original._image
                self._mask = original._mask
                self._loaded = True
            super().process(self._config)
        return self

    def unload(self) -> "RadiologyImage":
        self._image = None
        self._mask = None
        self._loaded = False
        self.history = []
        return self


# ----------------------------------------------------------------------
# worker entry points
# ----------------------------------------------------------------------
def _process_one_image(
    image: RadiologyImage,
    config: ProcessingConfig,
    out_dir: Optional[str],
    layout: str,
    skip_existing: bool,
) -> RadiologyImage:
    if out_dir is None:
        return image.copy().process(config)

    path = _output_path(image, out_dir, layout, config.output_format, config.compress)
    mask_path = _mask_output_path(path, layout)

    if skip_existing and os.path.exists(path):
        logger.debug("Skipping %s: %s already exists", image.series_id, path)
        return RadiologyImage(
            path,
            mask=mask_path if os.path.exists(mask_path) else None,
            series_id=image.series_id,
            metadata=image.metadata,
            lazy=True,
        )

    processed = image.copy().process(config)
    written = processed.save(
        path,
        mask_path=mask_path,
        output_format=config.output_format,
        compress=config.compress,
        save_mask=config.save_mask,
    )
    return RadiologyImage(
        written,
        mask=mask_path if os.path.exists(mask_path) else None,
        series_id=processed.series_id,
        metadata=processed.metadata,
        lazy=True,
    )


def _process_payload(payload: dict) -> dict:
    """Module-level worker so :class:`ProcessPoolExecutor` can pickle it."""
    config = ProcessingConfig.from_dict(payload["config"])
    image = RadiologyImage(
        payload["source"],
        mask=payload["mask_source"],
        series_id=payload["series_id"],
        metadata=payload["metadata"],
        masked_roi=payload["masked_roi"],
        lazy=True,
    )
    processed = _process_one_image(
        image, config, payload["out_dir"], payload["layout"], payload["skip_existing"]
    )
    return {
        "series_id": processed.series_id,
        "image_path": processed.source,
        "mask_path": processed.mask_source,
        "metadata": {
            key: value for key, value in processed.metadata.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }


# ----------------------------------------------------------------------
# path helpers
# ----------------------------------------------------------------------
def _output_path(
    image: RadiologyImage,
    out_dir: str,
    layout: str,
    output_format: str,
    compress: bool,
) -> str:
    from .io import resolve_output_path

    series_id = image.series_id or "image"
    if layout == "case_dirs":
        base = os.path.join(out_dir, series_id, "imaging")
    else:
        base = os.path.join(out_dir, series_id)
    return resolve_output_path(base, output_format=output_format, compress=compress)


def _mask_output_path(image_path: str, layout: str = "flat") -> str:
    """Where the mask goes for a given image path.

    ``case_dirs`` writes ``<case>/segmentation.nii.gz`` next to
    ``<case>/imaging.nii.gz``, which is the layout
    :meth:`Dataset.from_directory` reads back; ``flat`` appends ``_mask``.
    """
    for extension in (".nii.gz", ".nii", ".npy"):
        if image_path.lower().endswith(extension):
            stem = image_path[: -len(extension)]
            if layout == "case_dirs":
                return os.path.join(os.path.dirname(stem), "segmentation") + extension
            return stem + "_mask" + extension
    return image_path + "_mask"


def _scan_case_directories(root: str) -> list:
    """``root/<series_id>/imaging.nii.gz`` (+ optional mask)."""
    pairs = []
    for entry in sorted(os.listdir(root)):
        case_dir = os.path.join(root, entry)
        if not os.path.isdir(case_dir):
            continue
        names = set(os.listdir(case_dir))
        image_name = next((name for name in DEFAULT_IMAGE_NAMES if name in names), None)
        if image_name is None:
            continue
        mask_name = next((name for name in DEFAULT_MASK_NAMES if name in names), None)
        if mask_name is None:
            # Fall back to any file whose stem marks it as a mask, e.g.
            # "imaging_mask.nii.gz" from the flat output layout.
            from .io import _strip_image_suffix

            mask_name = next(
                (
                    name for name in sorted(names)
                    if name.lower().endswith(NIFTI_SUFFIXES)
                    and _strip_image_suffix(name).lower().endswith(MASK_SUFFIXES)
                ),
                None,
            )
        pairs.append((
            entry,
            os.path.join(case_dir, image_name),
            os.path.join(case_dir, mask_name) if mask_name else None,
        ))
    return pairs


def _scan_flat_files(root: str) -> list:
    """A directory of NIfTI files, pairing ``x.nii.gz`` with ``x_mask.nii.gz``."""
    from .io import _strip_image_suffix

    files = sorted(
        name for name in os.listdir(root)
        if name.lower().endswith(NIFTI_SUFFIXES) and os.path.isfile(os.path.join(root, name))
    )
    stems = {_strip_image_suffix(name): name for name in files}

    pairs = []
    for stem, name in stems.items():
        if any(stem.lower().endswith(suffix) for suffix in MASK_SUFFIXES):
            continue
        mask_name = next(
            (stems[stem + suffix] for suffix in MASK_SUFFIXES if stem + suffix in stems),
            None,
        )
        pairs.append((
            stem,
            os.path.join(root, name),
            os.path.join(root, mask_name) if mask_name else None,
        ))
    return pairs


def _scan_dicom_directories(root: str) -> list:
    """``root/<series_id>/*.dcm``."""
    pairs = []
    for entry in sorted(os.listdir(root)):
        case_dir = os.path.join(root, entry)
        if not os.path.isdir(case_dir):
            continue
        has_dicom = any(
            name.lower().endswith((".dcm", ".ima"))
            for _, _, names in os.walk(case_dir)
            for name in names
        )
        if has_dicom:
            pairs.append((entry, case_dir, None))
    return pairs


def _scan_with_patterns(
    root: str,
    image_pattern: Optional[str],
    mask_pattern: Optional[str],
    recursive: bool,
) -> list:
    from .io import _strip_image_suffix

    matches, mask_matches = [], []
    walker = os.walk(root) if recursive else [(root, [], os.listdir(root))]
    for directory, _, names in walker:
        for name in sorted(names):
            path = os.path.join(directory, name)
            if image_pattern and fnmatch.fnmatch(name, image_pattern):
                matches.append(path)
            elif mask_pattern and fnmatch.fnmatch(name, mask_pattern):
                mask_matches.append(path)

    by_directory = {}
    for path in mask_matches:
        by_directory.setdefault(os.path.dirname(path), []).append(path)

    pairs = []
    for path in sorted(matches):
        directory = os.path.dirname(path)
        series_id = (
            os.path.basename(directory)
            if os.path.abspath(directory) != os.path.abspath(root)
            else _strip_image_suffix(os.path.basename(path))
        )
        candidates = by_directory.get(directory) or []
        pairs.append((series_id, path, candidates[0] if candidates else None))
    return pairs


# ----------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------
def _progress(iterable, description: str, enabled: bool, total: Optional[int] = None):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a dependency, but be safe
        return iterable
    return tqdm(iterable, desc=description, total=total)


def _to_frame(records: list):
    import pandas as pd

    return pd.DataFrame(records)
