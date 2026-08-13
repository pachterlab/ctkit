"""Organ and tumor segmentation.

TotalSegmentator is a command line tool that reads and writes files, so it is
run inside a temporary directory that is deleted as soon as the masks are in
memory. Nothing is left on disk unless `output_dir` names somewhere to keep the
segmentation files.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional, Sequence

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)

#: The label convention used throughout the package: 0 background, 1 organ,
#: 2 tumor. Where they overlap, tumor wins.
BACKGROUND, ORGAN, TUMOR = 0, 1, 2


class TotalSegmentatorNotFound(RuntimeError):
    """Raised when the TotalSegmentator executable is not available."""


def totalsegmentator_available() -> bool:
    return shutil.which("TotalSegmentator") is not None


def segment_organs(
    image: nib.Nifti1Image,
    organs: Sequence[str],
    task: str = "total",
    fast: bool = False,
    remove_small_blobs: bool = True,
    fill_holes: bool = True,
    morphological_closing: bool = True,
    device: Optional[str] = None,
    modality: str = "CT",
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> nib.Nifti1Image:
    """Segment `organs` with TotalSegmentator and return their union as a mask.

    See :func:`segment_organ_components` for the arguments; this is the same
    call with the individual structures merged into a single binary mask.
    """
    components = segment_organ_components(
        image,
        organs,
        task=task,
        fast=fast,
        remove_small_blobs=remove_small_blobs,
        fill_holes=fill_holes,
        morphological_closing=morphological_closing,
        device=device,
        modality=modality,
        output_dir=output_dir,
        extra_args=extra_args,
    )
    return combine_organ_masks(list(components.values()))


def segment_organ_components(
    image: nib.Nifti1Image,
    organs: Sequence[str],
    task: str = "total",
    fast: bool = False,
    remove_small_blobs: bool = True,
    fill_holes: bool = True,
    morphological_closing: bool = True,
    device: Optional[str] = None,
    modality: str = "CT",
    output_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> dict:
    """Segment `organs` and return one binary mask per structure.

    Parameters
    ----------
    image:
        The volume to segment, in memory.
    organs:
        TotalSegmentator structure names, e.g. ``["kidney_left", "kidney_right"]``.
    task:
        TotalSegmentator task. ``"total"`` covers the common organs; MRI input
        is automatically routed to ``"total_mr"``.
    fast:
        Use the 3 mm model. Much faster, coarser boundaries.
    fill_holes, morphological_closing:
        Applied slice-by-slice to each structure, to close the small gaps the
        network leaves at organ boundaries.
    output_dir:
        Where TotalSegmentator writes its NIfTI files. By default a temporary
        directory that is deleted once the masks are in memory; give a path to
        keep them. It is created if it does not exist, and files with the same
        names are overwritten; one that survives the run is reported as
        possibly stale rather than silently read back. The returned masks are
        the same either way — note that they are cleaned in memory, so the
        files on disk are the raw network output.

    Returns
    -------
    dict
        Structure name to ``uint8`` mask, each sharing the geometry of `image`.
        Structures the model did not produce are omitted.
    """
    if not organs:
        raise ValueError("`organs` must name at least one structure to segment")
    if not totalsegmentator_available():
        raise TotalSegmentatorNotFound(
            "TotalSegmentator is not on the PATH. Install it with "
            "`pip install TotalSegmentator`, or supply your own mask instead of "
            "calling .segment()."
        )

    if modality.upper() in ("MR", "MRI") and task == "total":
        task = "total_mr"

    with tempfile.TemporaryDirectory(prefix="ctkit_seg_") as tmp:
        image_path = os.path.join(tmp, "input.nii.gz")
        if output_dir is None:
            segmentation_dir = os.path.join(tmp, "segmentations")
        else:
            segmentation_dir = os.fspath(output_dir)
            os.makedirs(segmentation_dir, exist_ok=True)
        nib.save(image, image_path)

        # A reused output directory can already hold masks under the names this
        # run writes; anything the run does not touch would be read back as if
        # it belonged to this image, so keep the timestamps to check against.
        existing = {}
        for organ in organs:
            path = os.path.join(segmentation_dir, f"{organ}.nii.gz")
            if os.path.exists(path):
                existing[organ] = os.path.getmtime(path)

        command = [
            "TotalSegmentator",
            "-i", image_path,
            "-o", segmentation_dir,
            "--task", task,
        ]
        # Asking for only the structures we need saves a large amount of time
        # and disk on the `total` task.
        if task == "total" and organs:
            command += ["--roi_subset"] + list(organs)
        if fast:
            command += ["--fast"]
        if remove_small_blobs:
            command += ["--remove_small_blobs"]
        if device:
            command += ["--device", device]
        if extra_args:
            command += list(extra_args)

        logger.info("Running: %s", " ".join(command))
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "TotalSegmentator failed "
                f"(exit {completed.returncode}).\n{completed.stderr.strip()[-2000:]}"
            )

        if output_dir is not None:
            logger.info("Segmentation files kept in %s", segmentation_dir)

        masks = {}
        for organ in organs:
            path = os.path.join(segmentation_dir, f"{organ}.nii.gz")
            if not os.path.exists(path):
                logger.warning(
                    "TotalSegmentator produced no mask for %r (task=%s). "
                    "Check that the structure name is valid for this task.",
                    organ, task,
                )
                continue
            if existing.get(organ) == os.path.getmtime(path):
                logger.warning(
                    "%s was already in %s and this run did not rewrite it; "
                    "it may be left over from a different image.",
                    os.path.basename(path), segmentation_dir,
                )
            organ_image = nib.load(path)
            masks[organ] = nib.Nifti1Image(
                np.asanyarray(organ_image.dataobj), organ_image.affine, organ_image.header
            )

    if not masks:
        raise RuntimeError(
            f"TotalSegmentator produced none of the requested structures: "
            f"{', '.join(organs)}"
        )

    if fill_holes or morphological_closing:
        masks = {
            name: clean_mask(
                mask, fill_holes=fill_holes, morphological_closing=morphological_closing
            )
            for name, mask in masks.items()
        }

    return masks


def combine_organ_masks(masks: Iterable[nib.Nifti1Image]) -> nib.Nifti1Image:
    """Union a collection of binary masks that share a geometry."""
    masks = list(masks)
    if not masks:
        raise ValueError("No masks to combine")

    reference = masks[0]
    for mask in masks[1:]:
        if mask.shape != reference.shape:
            raise ValueError(
                f"Mask shapes differ: {mask.shape} vs {reference.shape}"
            )
        if not np.allclose(mask.affine, reference.affine, atol=1e-4):
            raise ValueError("Mask affines differ; masks are not in the same space")

    union = np.logical_or.reduce(
        [np.asanyarray(mask.dataobj) > 0 for mask in masks]
    ).astype(np.uint8)
    return nib.Nifti1Image(union, reference.affine, reference.header)


def clean_mask(
    mask: nib.Nifti1Image,
    fill_holes: bool = True,
    morphological_closing: bool = True,
    structure_size: int = 3,
) -> nib.Nifti1Image:
    """Fill holes and close small gaps, slice by slice in the axial plane."""
    if not fill_holes and not morphological_closing:
        return mask

    data = np.asanyarray(mask.dataobj) > 0
    if data.ndim == 2:
        data = _clean_slice(data, fill_holes, morphological_closing, structure_size)
    else:
        for index in range(data.shape[2]):
            data[:, :, index] = _clean_slice(
                data[:, :, index], fill_holes, morphological_closing, structure_size
            )

    return nib.Nifti1Image(data.astype(np.uint8), mask.affine, mask.header)


def _clean_slice(
    plane: np.ndarray,
    fill_holes: bool,
    morphological_closing: bool,
    structure_size: int,
) -> np.ndarray:
    if fill_holes:
        plane = ndi.binary_fill_holes(plane)
    if morphological_closing:
        plane = ndi.binary_closing(
            plane, structure=np.ones((structure_size, structure_size))
        )
    return plane


def combine_organ_and_tumor(
    organ_mask: nib.Nifti1Image,
    tumor_mask: Optional[nib.Nifti1Image],
    restrict_organs_to_tumor: bool = True,
    organ_components: Optional[dict] = None,
) -> nib.Nifti1Image:
    """Merge an organ mask and a tumor mask into one labeled mask.

    Organ voxels become 1 and tumor voxels 2; where they overlap, tumor wins.

    When `restrict_organs_to_tumor` is set and `organ_components` maps
    structure names to individual masks, only the structures that actually
    contain tumor are kept — for a paired organ such as the kidneys this
    discards the healthy contralateral side, which would otherwise dominate
    the intensity statistics.
    """
    if tumor_mask is None:
        organ_data = (np.asanyarray(organ_mask.dataobj) > 0).astype(np.uint8)
        return nib.Nifti1Image(organ_data, organ_mask.affine, organ_mask.header)

    if tumor_mask.shape != organ_mask.shape:
        raise ValueError(
            f"Tumor mask shape {tumor_mask.shape} does not match organ mask shape "
            f"{organ_mask.shape}. Resample the tumor mask onto the image grid first."
        )

    tumor = np.asanyarray(tumor_mask.dataobj) > 0
    combined = np.zeros(tumor.shape, dtype=np.uint8)

    if restrict_organs_to_tumor and organ_components:
        overlapping = {
            name: mask
            for name, mask in organ_components.items()
            if np.any(tumor & (np.asanyarray(mask.dataobj) > 0))
        }
        if not overlapping:
            logger.warning(
                "The tumor mask does not overlap any segmented organ; keeping all organs."
            )
            overlapping = organ_components
        for mask in overlapping.values():
            combined[np.asanyarray(mask.dataobj) > 0] = ORGAN
    else:
        combined[np.asanyarray(organ_mask.dataobj) > 0] = ORGAN

    combined[tumor] = TUMOR
    return nib.Nifti1Image(combined, organ_mask.affine, organ_mask.header)


def binarize_masked_image(
    image: nib.Nifti1Image,
    label: int = TUMOR,
    tolerance: float = 1e-5,
) -> nib.Nifti1Image:
    """Recover a mask from an image that was *masked* rather than labeled.

    Some archives distribute a "ROI" volume that holds the original intensities
    inside the region of interest and the image minimum (air) everywhere else.
    That is an image, not a label map: thresholding just above the minimum
    turns it back into a mask.
    """
    data = np.asanyarray(image.dataobj)
    mask = (data > (data.min() + tolerance)).astype(np.uint8) * label
    return nib.Nifti1Image(mask, image.affine, image.header)


def resample_mask_to(
    mask: nib.Nifti1Image, reference: nib.Nifti1Image
) -> nib.Nifti1Image:
    """Put `mask` on the voxel grid of `reference` with nearest-neighbor."""
    import SimpleITK as sitk

    from .io import nifti_to_sitk, sitk_to_nifti

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(nifti_to_sitk(reference))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled = resampler.Execute(nifti_to_sitk(mask))
    out = sitk_to_nifti(resampled)
    return nib.Nifti1Image(
        np.asanyarray(out.dataobj).astype(np.uint8), reference.affine, reference.header
    )


def dice(mask_a: np.ndarray, mask_b: np.ndarray, label: Optional[int] = None) -> float:
    """Dice overlap between two masks, optionally restricted to one label."""
    if label is not None:
        first, second = (mask_a == label), (mask_b == label)
    else:
        first, second = (mask_a > 0), (mask_b > 0)

    total = first.sum() + second.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(first, second).sum() / total)
