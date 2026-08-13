"""Tests for the per-image processing steps."""

from __future__ import annotations

import os
import subprocess

import nibabel as nib
import numpy as np
import pytest

from ctkit import ProcessingConfig, RadiologyImage
from ctkit.io import nifti_to_sitk, sitk_to_nifti




class TestConstruction:
    def test_accepts_nifti_and_array(self, volume):
        data, mask = volume
        assert RadiologyImage(data).shape == data.shape
        assert RadiologyImage(np.zeros((4, 5, 6))).shape == (4, 5, 6)

    def test_accepts_sitk(self, volume):
        data, _ = volume
        image = RadiologyImage(nifti_to_sitk(data))
        assert image.shape == data.shape

    def test_accepts_path(self, case_dir):
        image = RadiologyImage(
            str(case_dir / "imaging.nii.gz"), mask=str(case_dir / "segmentation.nii.gz")
        )
        assert image.has_mask
        # The generic filename means the case directory names the series.
        assert image.series_id == "case_00"

    def test_lazy_loading_defers_read(self, case_dir):
        image = RadiologyImage(str(case_dir / "imaging.nii.gz"), lazy=True)
        assert image._image is None
        assert image.shape  # touching the data loads it
        assert image._image is not None

    def test_masked_roi_is_binarized(self, volume):
        data, _ = volume
        # An ROI file that stores intensities inside the region and air outside.
        roi_data = np.asanyarray(data.dataobj).copy()
        roi = nib.Nifti1Image(roi_data, data.affine)
        image = RadiologyImage(data, mask=roi, masked_roi=True)
        assert set(np.unique(image.mask_array)) <= {0, 2}
        assert (image.mask_array > 0).any()

    def test_rejects_nonsense(self):
        with pytest.raises(TypeError, match="Cannot load an image"):
            RadiologyImage(object())


class TestOrient:
    def test_reorients_to_ras(self, image):
        assert image.orientation != "RAS"
        image.orient()
        assert image.orientation == "RAS"

    def test_mask_follows_the_image(self, image):
        before = int((image.mask_array == 2).sum())
        image.orient()
        assert image.orientation == "RAS"
        assert int((image.mask_array == 2).sum()) == before
        assert image.mask.shape == image.shape

    def test_is_idempotent(self, image):
        image.orient().orient()
        assert image.orientation == "RAS"

    def test_non_ras_target(self, image):
        image.orient("LPS")
        assert image.orientation == "LPS"

    def test_rejects_bad_axis_codes(self, image):
        with pytest.raises(ValueError, match="Invalid orientation"):
            image.orient("XYZ")

    def test_preserves_world_coordinates(self, image):
        """Reorienting relabels the axes; it must not move the anatomy.

        The tumor's center of mass is used as the landmark because it is
        independent of the voxel storage order.
        """
        from scipy.ndimage import center_of_mass

        original = image.copy()
        world_before = nib.affines.apply_affine(
            original.affine, center_of_mass(original.mask_array == 2)
        )

        image.orient()
        world_after = nib.affines.apply_affine(
            image.affine, center_of_mass(image.mask_array == 2)
        )
        np.testing.assert_allclose(world_before, world_after, atol=1e-3)


class TestClip:
    def test_clamps_the_range(self, image):
        image.clip(-200, 300)
        assert image.array.min() >= -200
        assert image.array.max() <= 300

    def test_one_sided(self, image):
        image.clip(min_value=0, max_value=None)
        assert image.array.min() >= 0

    def test_requires_a_bound(self, image):
        with pytest.raises(ValueError, match="needs min_value"):
            image.clip(None, None)


class TestSegment:
    """TotalSegmentator is faked: the point is the file handling around it."""

    @pytest.fixture
    def fake_totalsegmentator(self, monkeypatch, volume):
        """Stand in for the executable, writing one mask per requested organ."""
        from ctkit import segmentation as seg

        data, _ = volume
        calls = []

        def run(command, **_):
            calls.append(list(command))
            output_dir = command[command.index("-o") + 1]
            os.makedirs(output_dir, exist_ok=True)
            organs = command[command.index("--roi_subset") + 1:]
            sphere = (np.asanyarray(data.dataobj) > 0).astype(np.uint8)
            for organ in organs:
                nib.save(
                    nib.Nifti1Image(sphere, data.affine),
                    os.path.join(output_dir, f"{organ}.nii.gz"),
                )
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(seg, "totalsegmentator_available", lambda: True)
        monkeypatch.setattr(seg.subprocess, "run", run)
        return calls

    def test_writes_nothing_by_default(self, fake_totalsegmentator, image):
        image.segment(organs=["kidney_left"], replace=True)
        assert image.has_mask
        output_dir = fake_totalsegmentator[0][fake_totalsegmentator[0].index("-o") + 1]
        assert not os.path.exists(output_dir)

    def test_output_dir_keeps_the_files(self, fake_totalsegmentator, image, tmp_path):
        destination = tmp_path / "segmentations"
        image.segment(organs=["kidney_left"], output_dir=str(destination), replace=True)
        assert (destination / "kidney_left.nii.gz").exists()

    def test_output_dir_is_created(self, fake_totalsegmentator, image, tmp_path):
        destination = tmp_path / "nested" / "segmentations"
        image.segment(organs=["kidney_left"], output_dir=destination, replace=True)
        assert (destination / "kidney_left.nii.gz").exists()

    def test_warns_about_a_file_the_run_did_not_rewrite(
        self, monkeypatch, image, tmp_path, caplog, volume
    ):
        from ctkit import segmentation as seg

        destination = tmp_path / "segmentations"
        destination.mkdir()
        data, _ = volume
        nib.save(data, str(destination / "kidney_left.nii.gz"))  # left over

        monkeypatch.setattr(seg, "totalsegmentator_available", lambda: True)
        monkeypatch.setattr(
            seg.subprocess,
            "run",
            lambda command, **_: subprocess.CompletedProcess(command, 0, "", ""),
        )

        with caplog.at_level("WARNING"):
            image.segment(
                organs=["kidney_left"], output_dir=str(destination), replace=True
            )
        assert "may be left over" in caplog.text

    def test_config_segmentation_dir_is_per_series(
        self, fake_totalsegmentator, image, tmp_path
    ):
        destination = tmp_path / "segmentations"
        image.process(
            ProcessingConfig(
                segment=True,
                organs=["kidney_left"],
                segmentation_dir=str(destination),
                standardize_size=False,
            )
        )
        assert (destination / "synthetic" / "kidney_left.nii.gz").exists()

    def test_cohort_gets_one_directory_per_series(
        self, fake_totalsegmentator, cohort_dir, tmp_path
    ):
        from ctkit import Dataset

        destination = tmp_path / "segmentations"
        data = Dataset(str(cohort_dir)).segment(
            organs=["kidney_left"], output_dir=str(destination), replace=True
        )
        series_ids = []
        for image in data:
            image.shape  # touching the data is what runs the deferred step
            series_ids.append(image.series_id)
        assert sorted(os.listdir(destination)) == sorted(series_ids)


class TestResample:
    def test_changes_spacing_and_shape(self, image):
        image.orient()
        original_shape = image.shape
        image.resample((0.8, 0.8, 3.0))
        np.testing.assert_allclose(image.spacing, (0.8, 0.8, 3.0), atol=1e-5)
        assert image.shape != original_shape

    def test_preserves_physical_extent(self, image):
        image.orient()
        extent_before = np.array(image.shape) * np.array(image.spacing)
        image.resample((0.8, 0.8, 3.0))
        extent_after = np.array(image.shape) * np.array(image.spacing)
        # Rounding to whole voxels allows a little slack.
        np.testing.assert_allclose(extent_before, extent_after, rtol=0.05)

    def test_mask_stays_a_label_map(self, image):
        image.orient().resample((1.0, 1.0, 2.0))
        assert set(np.unique(image.mask_array)) <= {0, 1, 2}
        assert image.mask.shape == image.shape

    def test_none_keeps_an_axis(self, image):
        image.orient()
        original = image.spacing
        image.resample((1.0, 1.0, None))
        assert image.spacing[2] == pytest.approx(original[2], abs=1e-5)

    def test_no_op_when_already_at_target(self, image):
        image.orient().resample((0.8, 0.8, 3.0))
        shape = image.shape
        image.resample((0.8, 0.8, 3.0))
        assert image.shape == shape


class TestApplyMask:
    def test_blanks_outside_and_crops(self, image):
        original_shape = image.shape
        image.apply_mask()
        assert image.shape < original_shape
        assert image.mask.shape == image.shape

    def test_selects_labels(self, image):
        image.apply_mask(labels=2, crop=False)
        kept = image.array > image.array.min()
        assert kept.sum() <= (image.mask_array == 2).sum()

    def test_padding_widens_the_box(self, image):
        tight = image.copy().apply_mask(padding=0)
        padded = image.copy().apply_mask(padding=5)
        assert all(wide >= narrow for wide, narrow in zip(padded.shape, tight.shape))

    def test_requires_a_mask(self, volume):
        with pytest.raises(ValueError, match="needs a mask"):
            RadiologyImage(volume[0]).apply_mask()

    def test_shape_mismatch_is_explained(self, volume):
        data, _ = volume
        wrong = nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), data.affine)
        image = RadiologyImage(data, mask=wrong)
        with pytest.raises(ValueError, match="does not match mask shape"):
            image.apply_mask()

    def test_cropping_preserves_world_coordinates(self, image):
        marker = np.array(np.unravel_index(
            int(np.argmax(image.mask_array == 2)), image.shape
        ))
        world_before = nib.affines.apply_affine(image.affine, marker)
        image.apply_mask()
        marker_after = np.array(np.unravel_index(
            int(np.argmax(image.mask_array == 2)), image.shape
        ))
        world_after = nib.affines.apply_affine(image.affine, marker_after)
        np.testing.assert_allclose(world_before, world_after, atol=1e-3)


class TestStandardizeSize:
    def test_pads_and_crops_to_target(self, image):
        image.standardize_size(64, 64, 10)
        assert image.shape == (64, 64, 10)
        assert image.mask.shape == (64, 64, 10)

    def test_none_leaves_an_axis_alone(self, image):
        depth = image.shape[2]
        image.standardize_size(30, 30, None)
        assert image.shape == (30, 30, depth)

    def test_keeps_the_center(self, image):
        """Center-cropping must keep the tumor, which sits at the center."""
        image.standardize_size(30, 30, 12)
        assert (image.mask_array == 2).any()

    def test_pad_value(self, image):
        """Padding fills with the requested value, not with zeros."""
        image.standardize_size(80, 80, None, fill_value=-1000)
        assert image.array[0, 0, 0] == pytest.approx(-1000, abs=1e-3)
        assert image.mask_array[0, 0, 0] == 0


class TestNormalize:
    def test_volume_method_gives_zero_mean(self, image):
        image.normalize()
        assert image.array.mean() == pytest.approx(0, abs=1e-4)
        assert image.array.std() == pytest.approx(1, abs=1e-4)

    def test_dataset_method_uses_supplied_statistics(self, image):
        original = image.array.copy()
        image.normalize(method="dataset", mean=100.0, std=50.0)
        np.testing.assert_allclose(image.array, (original - 100.0) / 50.0, rtol=1e-5)

    def test_dataset_method_needs_statistics(self, image):
        with pytest.raises(ValueError, match="pooled over the cohort"):
            image.normalize(method="dataset")

    def test_uniform_image_is_rejected(self):
        image = RadiologyImage(np.ones((8, 8, 8)))
        with pytest.raises(ValueError, match="uniform"):
            image.normalize()


class TestSelectSlice:
    def test_reduces_to_two_dimensions(self, image):
        image.select_slice(label=2)
        assert image.ndim == 2
        assert image.mask.shape == image.shape

    def test_picks_the_slice_with_most_label(self, image):
        counts = (image.mask_array == 2).sum(axis=(0, 1))
        expected = int(np.argmax(counts))
        image.select_slice(label=2)
        assert image.metadata["selected_slice"] == expected
        assert image.metadata["selected_slice_mask_voxels"] == int(counts[expected])

    def test_measures_several_labels_together(self, image):
        counts = np.isin(image.mask_array, [1, 2]).sum(axis=(0, 1))
        image.select_slice(label=[1, 2])
        assert image.metadata["selected_slice"] == int(np.argmax(counts))

    def test_missing_label_falls_back_to_slice_zero(self, image):
        image.select_slice(label=7)
        assert image.metadata["selected_slice"] == 0
        assert image.metadata["selected_slice_mask_voxels"] == 0
        assert image.ndim == 2

    def test_keepdims(self, image):
        image.select_slice(label=2, keepdims=True)
        assert image.ndim == 3 and image.shape[2] == 1

    def test_mask_mode_requires_a_mask(self, volume):
        with pytest.raises(ValueError, match="needs a mask"):
            RadiologyImage(volume[0]).select_slice()


class TestSelectSliceLabelDefault:
    """With no label given, a binary mask is unambiguous and a multi-label one is not."""

    def test_binary_mask_uses_its_only_label(self, volume):
        data, mask = volume
        binary = nib.Nifti1Image(
            (np.asanyarray(mask.dataobj) > 0).astype(np.uint8), mask.affine
        )
        image = RadiologyImage(data, mask=binary)
        counts = (np.asanyarray(binary.dataobj) == 1).sum(axis=(0, 1))

        image.select_slice()
        assert image.metadata["selected_slice_label"] == 1
        assert image.metadata["selected_slice"] == int(np.argmax(counts))

    def test_multi_label_mask_asks_for_a_label(self, image):
        with pytest.raises(ValueError, match="several labels"):
            image.select_slice()

    def test_empty_mask_keeps_slice_zero(self, volume):
        data, mask = volume
        empty = nib.Nifti1Image(np.zeros(mask.shape, dtype=np.uint8), mask.affine)
        image = RadiologyImage(data, mask=empty).select_slice()
        assert image.metadata["selected_slice"] == 0
        assert image.ndim == 2


class TestSelectSliceByIndex:
    def test_keeps_the_requested_slice(self, image):
        expected = image.array[:, :, 5].copy()
        image.select_slice(mode="index", index=5)
        assert image.metadata["selected_slice"] == 5
        np.testing.assert_array_equal(image.array, expected)

    def test_negative_index_counts_from_the_end(self, image):
        last = image.shape[2] - 1
        image.select_slice(mode="index", index=-1)
        assert image.metadata["selected_slice"] == last

    def test_needs_an_index(self, image):
        with pytest.raises(ValueError, match="needs index"):
            image.select_slice(mode="index")

    def test_rejects_an_out_of_range_index(self, image):
        with pytest.raises(IndexError, match="out of range"):
            image.select_slice(mode="index", index=999)

    def test_needs_no_mask(self, volume):
        image = RadiologyImage(volume[0]).select_slice(mode="index", index=3)
        assert image.ndim == 2

    def test_rejects_an_unknown_mode(self, image):
        with pytest.raises(ValueError, match="'mask' or 'index'"):
            image.select_slice(mode="tumor")


class TestProcess:
    def test_runs_the_configured_steps_in_order(self, image):
        config = ProcessingConfig(
            segment=False, target_shape=(40, 40, 12), normalize=True
        )
        image.process(config)
        assert image.applied_steps == [
            "orient", "clip", "resample", "apply_mask", "standardize_size", "normalize"
        ]
        assert image.shape == (40, 40, 12)

    def test_minimal_config_changes_nothing(self, image):
        original = image.array.copy()
        image.process(ProcessingConfig.minimal())
        np.testing.assert_array_equal(image.array, original)

    def test_overrides(self, image):
        image.process(ProcessingConfig.minimal(), clip=True, clip_min=-100, clip_max=100)
        assert image.array.min() >= -100

    def test_missing_mask_skips_masking_rather_than_failing(self, volume, caplog):
        image = RadiologyImage(volume[0])
        image.process(ProcessingConfig(segment=False, standardize_size=False))
        assert "apply_mask" not in image.applied_steps
        assert "no mask is available" in caplog.text

    def test_unresolved_target_shape_is_skipped_with_a_warning(self, image, caplog):
        image.process(ProcessingConfig(segment=False, target_shape=None))
        assert "standardize_size" not in image.applied_steps
        assert "no cohort" in caplog.text

    def test_two_dimensional_output_needs_a_mask(self, volume):
        image = RadiologyImage(volume[0])
        with pytest.raises(ValueError, match="requires a mask"):
            image.process(ProcessingConfig(segment=False, dimensionality="2D"))

    def test_two_dimensional_output_by_index_needs_no_mask(self, volume):
        image = RadiologyImage(volume[0])
        image.process(ProcessingConfig(
            segment=False, dimensionality="2D", slice_selection_mode="index",
            slice_index=4, mask=False, standardize_size=False,
        ))
        assert image.ndim == 2
        assert image.metadata["selected_slice"] == 4


class TestSaveAndCopy:
    def test_round_trip(self, image, tmp_path):
        image.process(ProcessingConfig(segment=False, target_shape=(32, 32, 8)))
        path = image.save(str(tmp_path / "out.nii.gz"))
        reloaded = RadiologyImage(path, mask=str(tmp_path / "out_mask.nii.gz"))
        assert reloaded.shape == image.shape
        np.testing.assert_allclose(reloaded.array, image.array, rtol=1e-5)
        np.testing.assert_array_equal(reloaded.mask_array, image.mask_array)

    def test_saves_only_the_requested_files(self, image, tmp_path):
        image.save(str(tmp_path / "out.nii.gz"))
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "out.nii.gz", "out_mask.nii.gz"
        ]

    def test_numpy_output(self, image, tmp_path):
        path = image.save(str(tmp_path / "out"), output_format="numpy")
        assert path.endswith(".npy")
        np.testing.assert_allclose(np.load(path), image.array, rtol=1e-5)

    def test_directory_target_uses_the_series_id(self, image, tmp_path):
        path = image.save(str(tmp_path) + "/")
        assert path.endswith("synthetic.nii.gz")

    def test_copy_is_independent(self, image):
        clone = image.copy()
        clone.clip(0, 1)
        assert image.array.max() > 1

    def test_history_records_the_chain(self, image):
        image.orient().clip(-200, 300)
        assert image.applied_steps == ["orient", "clip"]
        assert image.history[1]["min_value"] == -200


class TestGeometryConversions:
    def test_sitk_round_trip_preserves_geometry(self, volume):
        data, _ = volume
        restored = sitk_to_nifti(nifti_to_sitk(data))
        np.testing.assert_allclose(restored.affine, data.affine, atol=1e-5)
        np.testing.assert_allclose(
            np.asanyarray(restored.dataobj), np.asanyarray(data.dataobj), rtol=1e-5
        )

    def test_sitk_conversion_keeps_axis_order(self, volume):
        data, _ = volume
        assert nifti_to_sitk(data).GetSize() == data.shape

    def test_statistics(self, image):
        stats = image.statistics()
        assert stats["series_id"] == "synthetic"
        assert stats["mask_labels"] == [1, 2]
        assert stats["shape"] == image.shape
