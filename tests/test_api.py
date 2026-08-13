"""Tests for the functional API: ``import ctkit`` and call the steps."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

import ctkit
from ctkit import Dataset, ProcessingConfig, QCCriteria, RadiologyImage

from .conftest import make_volume

LENIENT = QCCriteria(min_slices=10)


class TestDispatch:
    def test_path_gives_one_image(self, case_dir):
        result = ctkit.orient(str(case_dir / "imaging.nii.gz"))
        assert isinstance(result, RadiologyImage)
        assert result.orientation == "RAS"

    def test_directory_gives_a_cohort(self, cohort_dir):
        result = ctkit.orient(str(cohort_dir))
        assert isinstance(result, Dataset)
        assert len(result) == 6

    def test_array_gives_one_image(self):
        result = ctkit.clip(np.full((8, 8, 8), 500.0), -200, 300)
        assert isinstance(result, RadiologyImage)
        assert result.array.max() == 300

    def test_list_gives_a_cohort(self, cohort_dir):
        paths = [str(case / "imaging.nii.gz") for case in sorted(cohort_dir.iterdir())]
        assert len(Dataset(paths)) == len(paths)

    def test_csv_gives_a_cohort(self, cohort_dir, tmp_path):
        import pandas as pd

        rows = [
            {"series_id": case.name, "Image": str(case / "imaging.nii.gz")}
            for case in sorted(cohort_dir.iterdir())
        ]
        table = tmp_path / "metadata.csv"
        pd.DataFrame(rows).to_csv(table, index=False)
        assert len(Dataset(str(table))) == len(rows)


class TestConstructor:
    """``Dataset(...)`` is the one entry point for a cohort."""

    def test_directory(self, cohort_dir):
        data = Dataset(str(cohort_dir))
        assert len(data) == 6
        assert data.name == cohort_dir.name
        assert data.get("case_00").mask_source is not None

    def test_glob(self, cohort_dir):
        assert len(Dataset(str(cohort_dir / "*" / "imaging.nii.gz"))) == 6

    def test_single_file(self, case_dir):
        assert len(Dataset(str(case_dir / "imaging.nii.gz"))) == 1

    def test_an_array_is_one_series_not_a_stack_of_slices(self):
        data, _ = make_volume(shape=(16, 16, 8))

        assert len(Dataset(np.asanyarray(data.dataobj))) == 1
        assert len(Dataset(data)) == 1

    def test_a_string_is_not_iterated_into_characters(self, cohort_dir):
        assert Dataset(str(cohort_dir)).series_ids != list(str(cohort_dir))

    def test_a_missing_path_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            Dataset(str(tmp_path / "absent"))

    def test_an_empty_glob_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files match"):
            Dataset(str(tmp_path / "*.nii.gz"))

    def test_a_dataset_is_copied_not_nested(self, cohort_dir):
        source = Dataset(str(cohort_dir))
        assert Dataset(source).series_ids == source.series_ids


class TestChaining:
    """The methods on a cohort mirror the methods on an image."""

    def test_the_whole_chain(self, cohort_dir, tmp_path):
        out = tmp_path / "processed"

        written = (
            Dataset(str(cohort_dir))
            .filter(min_slices=10, progress=False)
            .orient()
            .clip(-200, 300)
            .apply_mask(padding=2)
            .standardize_size(20, 20, 10)
            .save(str(out), progress=False)
        )

        assert len(written) == 5
        assert Dataset(str(out))[0].shape == (20, 20, 10)

    def test_a_method_matches_its_function(self, cohort_dir):
        by_method = Dataset(str(cohort_dir)).clip(-100, 100)[0]
        by_function = ctkit.clip(str(cohort_dir), -100, 100)[0]
        assert np.array_equal(by_method.array, by_function.array)

    def test_check_reports_over_a_cohort(self, cohort_dir):
        report = Dataset(str(cohort_dir)).check(min_slices=10, progress=False)
        assert report["passed"].sum() == 5

    def test_filter_is_not_an_image_method(self, image):
        with pytest.raises(AttributeError, match="use check"):
            image.filter()


class TestOneImage:
    def test_functions_match_the_methods(self, image):
        expected = image.copy().clip(-200, 300).array

        returned = ctkit.clip(image, -200, 300)

        assert returned is image, "the functional API modifies in place, like the methods"
        assert np.array_equal(image.array, expected)

    def test_steps_compose(self, image):
        ctkit.orient(image)
        ctkit.clip(image, -200, 300)
        ctkit.resample(image, (1.0, 1.0, 3.0))
        ctkit.apply_mask(image, labels=[1, 2], padding=2)
        ctkit.standardize_size(image, 24, 24, 8)

        assert image.shape == (24, 24, 8)
        assert [step["step"] for step in image.history] == [
            "orient", "clip", "resample", "apply_mask", "standardize_size",
        ]

    def test_select_slice_takes_its_own_arguments(self, image):
        ctkit.select_slice(image, label=2)
        assert image.ndim == 2

    def test_orient_target_is_not_shadowed(self, image):
        assert ctkit.orient(image, target="LPS").orientation == "LPS"

    def test_normalize(self, image):
        ctkit.normalize(image)
        assert image.array.mean() == pytest.approx(0.0, abs=1e-5)

    def test_save(self, image, tmp_path):
        out = tmp_path / "case.nii.gz"
        ctkit.save(image, str(out))
        assert out.exists()


class TestCohort:
    def test_steps_are_deferred_until_the_image_is_read(self, cohort_dir):
        dataset = Dataset(str(cohort_dir))
        clipped = ctkit.clip(dataset, -100, 100)

        assert clipped is not dataset
        assert all(not image._loaded for image in clipped.images)
        assert clipped[0].array.max() <= 100
        assert dataset[0].array.max() > 100, "the source cohort is untouched"

    def test_steps_compose_over_a_cohort(self, cohort_dir):
        chain = ctkit.standardize_size(
            ctkit.apply_mask(ctkit.clip(str(cohort_dir), -100, 100), padding=2),
            20, 20, 10,
        )
        image = chain.get("case_00").load()

        assert image.shape == (20, 20, 10)
        assert image.array.max() <= 100
        assert [step["step"] for step in image.history] == [
            "clip", "apply_mask", "standardize_size",
        ]

    def test_deferred_steps_survive_unloading(self, cohort_dir):
        chain = ctkit.clip(ctkit.orient(str(cohort_dir)), -100, 100)
        image = chain.get("case_00")

        first = image.load().array.max()
        image.unload()

        assert image.load().array.max() == first
        assert [step["step"] for step in image.history] == ["orient", "clip"]

    def test_in_memory_images_are_modified_in_place(self):
        data, _ = make_volume(shape=(16, 16, 8))
        dataset = Dataset([data])

        result = ctkit.clip(dataset, -100, 100)

        assert result[0] is dataset[0]
        assert dataset[0].array.max() <= 100

    def test_saving_a_cohort_runs_the_deferred_steps(self, cohort_dir, tmp_path):
        out = tmp_path / "processed"
        ctkit.save(ctkit.clip(str(cohort_dir), -100, 100), str(out), progress=False)

        written = Dataset.from_directory(str(out))
        assert len(written) == 6
        assert written[0].array.max() <= 100

    def test_deferred_steps_survive_a_parallel_run(self, cohort_dir, tmp_path):
        """Workers rebuild an image from its path, so a staged step must not be lost."""
        staged = ctkit.clip(str(cohort_dir), -100, 100)
        passthrough = ProcessingConfig(
            segment=False, clip=False, resample=False, mask=False, standardize_size=False
        )

        result = ctkit.process(
            staged, passthrough, out_dir=str(tmp_path / "out"), workers=2, progress=False
        )

        assert result[0].array.max() <= 100

    def test_apply_rejects_an_unknown_step(self, cohort_dir):
        with pytest.raises(AttributeError, match="not a RadiologyImage method"):
            Dataset(str(cohort_dir)).apply("sharpen")


class TestQualityControl:
    def test_filter_a_directory(self, cohort_dir):
        usable = ctkit.filter(str(cohort_dir), min_slices=10, progress=False)

        assert len(usable) == 5
        assert "case_bad" not in usable.series_ids
        assert len(usable.qc_report) == 6
        assert len(usable.rejected) == 1

    def test_presets(self, cohort_dir):
        assert len(ctkit.filter(str(cohort_dir), "permissive", progress=False)) == 6
        assert len(ctkit.filter(str(cohort_dir), progress=False)) == 0

    def test_thresholds_override_a_criteria_object(self, cohort_dir):
        usable = ctkit.filter(
            str(cohort_dir), QCCriteria.permissive(), min_slices=17, progress=False
        )
        assert len(usable) < 5

    def test_unknown_threshold_is_reported(self, cohort_dir):
        with pytest.raises(TypeError, match="min_slice"):
            ctkit.filter(str(cohort_dir), min_slice=10, progress=False)

    def test_unknown_preset_is_reported(self, cohort_dir):
        with pytest.raises(ValueError, match="'default', 'radiomics' or 'permissive'"):
            ctkit.filter(str(cohort_dir), "lenient", progress=False)

    def test_check_one_image(self, image):
        result = ctkit.check(image, min_slices=10)
        assert result.passed
        assert result.stats

    def test_check_a_metadata_row(self):
        assert ctkit.check({"Modality": "CT", "SliceThickness": 2.5}).passed
        assert not ctkit.check({"Modality": "MR"}).passed

    def test_check_a_cohort_reports_without_dropping(self, cohort_dir):
        report = ctkit.check(str(cohort_dir), min_slices=10, progress=False)
        assert len(report) == 6
        assert report["passed"].sum() == 5


class TestProtocols:
    def test_process_one_image(self, case_dir):
        config = ProcessingConfig(segment=False, target_shape=(24, 24, 8))
        result = ctkit.process(str(case_dir / "imaging.nii.gz"), config)
        assert result.shape == (24, 24, 8)

    def test_process_a_cohort(self, cohort_dir, tmp_path):
        config = ProcessingConfig(segment=False, target_shape=(24, 24, 8))
        out = tmp_path / "processed"

        result = ctkit.process(
            ctkit.filter(str(cohort_dir), min_slices=10, progress=False),
            config,
            out_dir=str(out),
            progress=False,
        )

        assert len(result) == 5
        assert result[0].shape == (24, 24, 8)

    def test_process_by_collection_name(self, case_dir):
        result = ctkit.process(
            str(case_dir / "imaging.nii.gz"), "tcga-kirc", segment=False, standardize_size=False
        )
        assert result.array.min() >= -200

    def test_methods_take_a_protocol_by_name_too(self, case_dir, cohort_dir):
        image = RadiologyImage(str(case_dir / "imaging.nii.gz"))
        image.process("tcga-kirc", segment=False, mask=False, standardize_size=False)
        assert image.array.min() >= -200

        data = Dataset(str(cohort_dir)).process(
            "tcga-kirc", segment=False, mask=False, standardize_size=False, progress=False
        )
        assert data[0].array.min() >= -200

    def test_resolve_rejects_what_it_cannot_read(self):
        with pytest.raises(TypeError, match="Cannot read a protocol"):
            ProcessingConfig.resolve(42)

    def test_process_from_a_saved_protocol(self, case_dir, tmp_path):
        protocol = tmp_path / "protocol.yaml"
        ProcessingConfig(segment=False, target_shape=(24, 24, 8)).to_yaml(str(protocol))

        result = ctkit.process(str(case_dir / "imaging.nii.gz"), str(protocol))
        assert result.shape == (24, 24, 8)
