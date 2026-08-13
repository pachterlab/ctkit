"""Tests for configuration, quality control, I/O and feature extraction."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import pytest

from ctkit import ProcessingConfig, QCCriteria, RadiologyImage
from ctkit.datasets import get_dataset_info, list_datasets
from ctkit.io import (
    infer_series_id,
    load_image,
    resolve_output_path,
    save_image,
)
from ctkit.qc import check_series_metadata, check_volume

from .conftest import requires_radiomics


class TestConfig:
    def test_defaults_are_the_documented_protocol(self):
        config = ProcessingConfig()
        assert config.steps == [
            "orient", "clip", "resample", "apply_mask", "standardize_size"
        ]

    def test_for_dataset_pulls_curated_settings(self):
        config = ProcessingConfig.for_dataset("tcga-kirc")
        assert config.organs == ["kidney_left", "kidney_right"]
        assert (config.clip_min, config.clip_max) == (-200, 300)
        assert config.target_shape == (185, 185, 75)
        assert config.dataset == "tcga-kirc"

    def test_for_dataset_accepts_alternative_spellings(self):
        for name in ("TCGA-KIRC", "tcga_kirc", "TCGA KIRC"):
            assert ProcessingConfig.for_dataset(name).dataset == "tcga-kirc"

    def test_for_dataset_unmasked_uses_the_other_dimensions(self):
        config = ProcessingConfig.for_dataset("tcga-kirc", masked=False)
        assert config.target_shape == (625, 625, 200)
        assert config.mask is False

    def test_unknown_dataset_lists_the_options(self):
        with pytest.raises(KeyError, match="Known datasets"):
            ProcessingConfig.for_dataset("tcga-nope")

    def test_lung_window_differs_from_soft_tissue(self):
        assert ProcessingConfig.for_dataset("tcga-luad").clip_min == -1000
        assert ProcessingConfig.for_dataset("tcga-kirc").clip_min == -200

    def test_radiomics_preset_disables_duplicated_steps(self):
        config = ProcessingConfig.radiomics("tcga-kirc")
        assert not config.resample and not config.normalize and not config.mask
        assert config.orient and config.clip

    def test_minimal_preset_is_a_no_op(self):
        assert ProcessingConfig.minimal().steps == []

    def test_yaml_round_trip(self, tmp_path):
        config = ProcessingConfig.for_dataset("tcga-lihc")
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)
        assert ProcessingConfig.from_yaml(path).to_dict() == config.to_dict()

    def test_replace_returns_a_new_config(self):
        config = ProcessingConfig()
        modified = config.replace(clip_min=-500)
        assert modified.clip_min == -500
        assert config.clip_min == -200

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            ProcessingConfig.from_dict({"clip_min": 0, "nonsense": 1})

    def test_validation_catches_inconsistent_settings(self):
        with pytest.raises(ValueError, match="clip=True requires"):
            ProcessingConfig(clip=True, clip_min=None, clip_max=None)
        with pytest.raises(ValueError, match="segment=True requires"):
            ProcessingConfig(segment=True, organs=None)
        with pytest.raises(ValueError, match="dimensionality"):
            ProcessingConfig(dimensionality="4D")

    def test_needs_dataset_pass(self):
        assert ProcessingConfig(target_shape=None).needs_dataset_pass
        assert not ProcessingConfig(target_shape=(1, 1, 1)).needs_dataset_pass
        assert ProcessingConfig(
            target_shape=(1, 1, 1), normalize=True, normalization_method="dataset"
        ).needs_dataset_pass

    def test_describe_reads_as_a_protocol(self):
        text = ProcessingConfig.for_dataset("tcga-kirc").describe()
        assert "reorient to canonical RAS" in text
        assert "clip intensities to [-200, 300] HU" in text


class TestQualityControl:
    def test_accepts_a_normal_volume(self, volume):
        assert check_volume(volume[0], QCCriteria(min_slices=10))

    def test_rejects_too_few_slices(self, volume):
        result = check_volume(volume[0], QCCriteria(min_slices=100))
        assert not result
        assert "slices" in result.reason

    def test_rejects_4d(self):
        image = nib.Nifti1Image(np.zeros((8, 8, 8, 3)), np.eye(4))
        result = check_volume(image, QCCriteria(min_slices=1))
        assert not result
        assert "4D" in result.reason

    def test_rejects_extreme_spacing(self):
        image = nib.Nifti1Image(np.zeros((8, 8, 8)), np.diag([1, 1, 50, 1]))
        result = check_volume(image, QCCriteria(min_slices=1, max_spacing=20))
        assert not result and "spacing" in result.reason

    def test_rejects_anisotropic_in_plane_sampling(self):
        image = nib.Nifti1Image(np.zeros((8, 8, 8)), np.diag([1, 9, 2, 1]))
        result = check_volume(image, QCCriteria(min_slices=1, max_in_plane_anisotropy=4))
        assert not result and "anisotropy" in result.reason

    def test_collects_statistics_even_when_passing(self, volume):
        result = check_volume(volume[0], QCCriteria(min_slices=10))
        assert result.stats["n_slices"] == 20
        assert result.stats["orientation"] == "LPS"
        assert "in_plane_anisotropy" in result.stats

    def test_metadata_filter_drops_localizers(self):
        row = {"SeriesDescription": "Topogram 0.6 T20s", "Modality": "CT", "ImageCount": 1}
        result = check_series_metadata(row, QCCriteria())
        assert not result
        assert "topogram" in result.reason

    def test_metadata_filter_drops_thick_slices(self):
        row = {"SeriesDescription": "ABDOMEN", "Modality": "CT",
               "SliceThickness": 15.0, "ImageCount": 100}
        result = check_series_metadata(row, QCCriteria())
        assert not result and "thickness" in result.reason

    def test_metadata_filter_drops_other_modalities(self):
        row = {"SeriesDescription": "T2", "Modality": "MR", "ImageCount": 100}
        assert not check_series_metadata(row, QCCriteria(modality="CT"))

    def test_metadata_filter_keeps_a_diagnostic_series(self):
        row = {"SeriesDescription": "ARTERIAL PHASE 2.0 B31f", "Modality": "CT",
               "ImageCount": 251, "SliceThickness": 2.0}
        assert check_series_metadata(row, QCCriteria())

    def test_sharp_kernels_only_excluded_for_radiomics(self):
        row = {"SeriesDescription": "ABDOMEN 2.0 B70f", "Modality": "CT", "ImageCount": 200}
        assert check_series_metadata(row, QCCriteria())
        assert not check_series_metadata(row, QCCriteria.for_radiomics())

    def test_result_is_falsy_and_explains_itself(self):
        result = check_series_metadata(
            {"SeriesDescription": "scout", "Modality": "CT", "ImageCount": 1}, QCCriteria()
        )
        assert not bool(result)
        assert len(result.reasons) >= 1
        assert "passed" in result.to_dict()


class TestIO:
    def test_load_from_array_with_spacing(self):
        image = load_image(np.zeros((4, 5, 6)), spacing=(2.0, 2.0, 5.0))
        np.testing.assert_allclose(image.header.get_zooms()[:3], (2.0, 2.0, 5.0))

    def test_load_npy(self, tmp_path):
        path = str(tmp_path / "volume.npy")
        np.save(path, np.ones((3, 4, 5)))
        assert load_image(path).shape == (3, 4, 5)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/scan.nii.gz")

    @pytest.mark.parametrize(
        "path,output_format,compress,expected",
        [
            ("out", "nifti", True, "out.nii.gz"),
            ("out.nii", "nifti", True, "out.nii.gz"),
            ("out.nii.gz", "nifti", False, "out.nii"),
            ("out.nii.gz", "numpy", True, "out.npy"),
        ],
    )
    def test_output_extension(self, path, output_format, compress, expected):
        assert resolve_output_path(path, output_format, compress) == expected

    def test_infer_series_id_prefers_the_case_directory(self):
        assert infer_series_id("/data/TCGA-KM-8438/imaging.nii.gz") == "TCGA-KM-8438"
        assert infer_series_id("/data/scan_01.nii.gz") == "scan_01"

    def test_save_creates_parent_directories(self, tmp_path, volume):
        path = save_image(volume[0], str(tmp_path / "deep" / "nested" / "out.nii.gz"))
        assert os.path.exists(path)


class TestCatalogue:
    def test_lists_curated_and_extra_datasets(self):
        frame = list_datasets()
        assert "tcga-kirc" in set(frame["name"])
        assert "lidc-idri" in set(frame["name"])

    def test_lookup_by_collection_name(self):
        assert get_dataset_info("NSCLC-Radiomics")["name"] == "nsclc-radiomics"

    def test_unknown_dataset_suggests_alternatives(self):
        with pytest.raises(KeyError, match="Did you mean|Unknown dataset"):
            get_dataset_info("kirc-tcga-nonsense")


@requires_radiomics
class TestFeatures:
    def test_extracts_features(self, image):
        image.process(ProcessingConfig(segment=False, target_shape=(32, 32, 8)))
        features = image.radiomics(labels=[1, 2])
        assert features["series_id"] == "synthetic"
        real = [key for key in features if key.startswith("original_")]
        assert len(real) > 50

    def test_label_selection_changes_the_region(self, image):
        image.process(ProcessingConfig(segment=False, standardize_size=False))
        both = image.radiomics(labels=[1, 2])
        tumor_only = image.radiomics(labels=[2])
        assert (
            both["original_shape_MeshVolume"] > tumor_only["original_shape_MeshVolume"]
        )

    def test_requires_a_mask(self, volume):
        with pytest.raises(ValueError, match="needs a mask"):
            RadiologyImage(volume[0]).radiomics()

    def test_absent_labels_are_reported(self, image):
        with pytest.raises(ValueError, match="none of the labels"):
            image.radiomics(labels=[7, 8])
