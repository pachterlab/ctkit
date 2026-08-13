"""Tests for cohort-level processing."""

from __future__ import annotations



import nibabel as nib
import numpy as np
import pytest

from ctkit import Dataset, ProcessingConfig, QCCriteria

from .conftest import make_volume

LENIENT = QCCriteria(min_slices=10)


class TestDiscovery:
    def test_case_directories(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        assert len(dataset) == 6
        assert "case_00" in dataset.series_ids
        assert dataset.get("case_00").mask_source is not None

    def test_flat_files(self, tmp_path):
        data, mask = make_volume()
        nib.save(data, str(tmp_path / "scan_a.nii.gz"))
        nib.save(mask, str(tmp_path / "scan_a_mask.nii.gz"))
        nib.save(data, str(tmp_path / "scan_b.nii.gz"))

        dataset = Dataset.from_directory(str(tmp_path))
        assert sorted(dataset.series_ids) == ["scan_a", "scan_b"]
        assert dataset.get("scan_a").mask_source is not None
        assert dataset.get("scan_b").mask_source is None

    def test_explicit_patterns(self, tmp_path):
        data, mask = make_volume()
        case = tmp_path / "patient_1"
        case.mkdir()
        nib.save(data, str(case / "scan_CT.nii.gz"))
        nib.save(mask, str(case / "roi_label.nii.gz"))

        dataset = Dataset.from_directory(
            str(tmp_path), image_pattern="*_CT.nii.gz", mask_pattern="roi_*.nii.gz"
        )
        assert len(dataset) == 1
        assert dataset[0].series_id == "patient_1"
        assert dataset[0].has_mask

    def test_from_paths(self, cohort_dir):
        images = sorted(str(p / "imaging.nii.gz") for p in cohort_dir.iterdir())
        dataset = Dataset.from_paths(images)
        assert len(dataset) == len(images)

    def test_from_paths_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="they must match"):
            Dataset.from_paths(["a.nii.gz", "b.nii.gz"], masks=["a_mask.nii.gz"])

    def test_from_metadata(self, cohort_dir, tmp_path):
        import pandas as pd

        rows = [
            {
                "series_id": case.name,
                "Image": str(case / "imaging.nii.gz"),
                "Mask": str(case / "segmentation.nii.gz"),
                "Manufacturer": "SIEMENS",
            }
            for case in sorted(cohort_dir.iterdir())
            if (case / "segmentation.nii.gz").exists()
        ]
        csv_path = tmp_path / "metadata.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        dataset = Dataset.from_metadata(str(csv_path))
        assert len(dataset) == 5
        assert dataset[0].metadata["Manufacturer"] == "SIEMENS"

    def test_empty_directory_explains_itself(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No images found"):
            Dataset.from_directory(str(tmp_path))

    def test_slicing_and_lookup(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        assert isinstance(dataset[:2], Dataset)
        assert len(dataset[:2]) == 2
        with pytest.raises(KeyError):
            dataset.get("nope")


class TestFilter:
    def test_drops_unusable_series(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        kept = dataset.filter(LENIENT, progress=False)
        assert len(kept) == 5
        assert "case_bad" not in kept.series_ids

    def test_report_covers_every_series(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        kept = dataset.filter(LENIENT, progress=False)
        report = kept.qc_report
        assert len(report) == 6
        assert set(report.columns) >= {"series_id", "passed", "reason", "n_slices"}
        failure = report[report["series_id"] == "case_bad"].iloc[0]
        assert not failure["passed"]
        assert "3 slices" in failure["reason"]

    def test_permissive_criteria_keep_everything(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        assert len(dataset.filter(QCCriteria.permissive(), progress=False)) == 6

    def test_rejected_images_are_kept_for_disposal(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        kept = dataset.filter(LENIENT, progress=False)
        assert [image.series_id for image in kept.rejected] == ["case_bad"]

    def test_metadata_level_reads_no_volumes(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        kept = dataset.filter(LENIENT, level="metadata", progress=False)
        assert len(kept) == 6  # nothing in the headers disqualifies these
        assert "n_slices" not in kept.qc_report.columns

    def test_metadata_level_uses_the_metadata_rows(self, cohort_dir):
        import pandas as pd

        from ctkit.dataset import Dataset as _Dataset

        frame = pd.DataFrame([
            {"series_id": "case_00", "Image": str(cohort_dir / "case_00" / "imaging.nii.gz"),
             "Modality": "CT", "SeriesDescription": "ARTERIAL 2.0"},
            {"series_id": "case_01", "Image": str(cohort_dir / "case_01" / "imaging.nii.gz"),
             "Modality": "CT", "SeriesDescription": "SCOUT"},
        ])
        dataset = _Dataset.from_metadata(frame)
        kept = dataset.filter(LENIENT, level="metadata", progress=False)
        assert kept.series_ids == ["case_00"]

    def test_rejects_an_unknown_level(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        with pytest.raises(ValueError, match="'metadata', 'volume' or 'all'"):
            dataset.filter(LENIENT, level="headers", progress=False)


class TestDiscard:
    def test_moves_a_case_directory(self, cohort_dir, tmp_path):
        from ctkit.dataset import discard

        dataset = Dataset.from_directory(str(cohort_dir))
        rejected = dataset.filter(LENIENT, progress=False).rejected
        moved = discard(rejected, destination=str(tmp_path / "excluded"))

        assert len(moved) == 1
        assert (tmp_path / "excluded" / "case_bad" / "imaging.nii.gz").exists()
        assert not (cohort_dir / "case_bad").exists()

    def test_deletes_when_asked(self, cohort_dir):
        from ctkit.dataset import discard

        dataset = Dataset.from_directory(str(cohort_dir))
        discard(dataset.filter(LENIENT, progress=False).rejected, delete=True)
        assert not (cohort_dir / "case_bad").exists()
        assert (cohort_dir / "case_00" / "imaging.nii.gz").exists()

    def test_needs_a_destination_or_delete(self, cohort_dir):
        from ctkit.dataset import discard

        with pytest.raises(ValueError, match="either a destination or delete"):
            discard(Dataset.from_directory(str(cohort_dir)).images)

    def test_leaves_in_memory_images_alone(self, image):
        from ctkit.dataset import discard, files_of

        assert files_of(image) == []
        assert discard([image], delete=True) == []


class TestCohortStatistics:
    def test_shape_percentile(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        shape = dataset.shape_percentile(95, progress=False)
        assert len(shape) == 3
        # Between the smallest and largest case in the cohort.
        assert 40 <= shape[0] <= 56

    def test_intensity_statistics_match_a_direct_computation(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        stats = dataset.intensity_statistics(progress=False)

        pooled = np.concatenate([image.array.ravel() for image in dataset])
        assert stats["mean"] == pytest.approx(float(pooled.mean()), rel=1e-6)
        assert stats["std"] == pytest.approx(float(pooled.std()), rel=1e-6)
        assert stats["n_voxels"] == pooled.size

    def test_statistics_frame(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        frame = dataset.statistics(progress=False)
        assert len(frame) == 5
        assert {"series_id", "mean", "std", "shape"} <= set(frame.columns)


class TestStagedSplit:
    def test_per_image_steps_run_once_in_the_first_pass(self, cohort_dir):
        """A cohort-level protocol must not repeat crop_to_content in the tail."""
        config = ProcessingConfig(
            segment=False, crop_to_content=True, standardize_size=True
        )
        head, tail = Dataset.from_directory(str(cohort_dir))._split_config(config)

        assert "crop_to_content" in head.steps
        assert "crop_to_content" not in tail.steps


class TestProcess:
    def test_writes_only_final_files(self, cohort_dir, tmp_path):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        out_dir = tmp_path / "processed"
        dataset.process(
            ProcessingConfig(segment=False, target_shape=(32, 32, 8)),
            out_dir=str(out_dir),
            progress=False,
            on_error="raise",
        )
        for case in out_dir.iterdir():
            if case.is_dir():
                assert sorted(p.name for p in case.iterdir()) == [
                    "imaging.nii.gz", "segmentation.nii.gz"
                ]

    def test_output_is_rediscoverable(self, cohort_dir, tmp_path):
        """What process() writes must be what from_directory() can read back."""
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        out_dir = tmp_path / "processed"
        dataset.process(
            ProcessingConfig(segment=False, target_shape=(32, 32, 8)),
            out_dir=str(out_dir), progress=False, on_error="raise",
        )
        reloaded = Dataset.from_directory(str(out_dir))
        assert len(reloaded) == 5
        assert all(image.mask_source is not None for image in reloaded)
        assert all(image.shape == (32, 32, 8) for image in reloaded)

    def test_resolves_target_shape_from_the_cohort(self, cohort_dir, tmp_path):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        config = ProcessingConfig(segment=False, target_shape=None)
        assert config.needs_dataset_pass

        processed = dataset.process(
            config, out_dir=str(tmp_path / "out"), progress=False, on_error="raise"
        )
        shapes = {image.shape for image in processed}
        assert len(shapes) == 1, "every case should end up the same shape"

    def test_resolves_dataset_normalization(self, cohort_dir, tmp_path):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        config = ProcessingConfig(
            segment=False,
            target_shape=(32, 32, 8),
            normalize=True,
            normalization_method="dataset",
        )
        processed = dataset.process(
            config, out_dir=str(tmp_path / "out"), progress=False, on_error="raise"
        )
        pooled = np.concatenate([image.array.ravel() for image in processed])
        # Pooled over the cohort the mean is ~0; individual volumes are not.
        assert pooled.mean() == pytest.approx(0, abs=0.15)
        individual_means = [float(image.array.mean()) for image in processed]
        assert np.std(individual_means) > 1e-6

    def test_staging_matches_recomputing(self, cohort_dir, tmp_path):
        """The two ways of handling cohort-level steps must agree exactly.

        Staging runs the per-image steps once into a temporary directory;
        recomputing runs them again for each measurement pass. They are an
        optimization of each other, so the outputs have to be identical.
        """
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        config = ProcessingConfig(
            segment=False,
            target_shape=None,
            normalize=True,
            normalization_method="dataset",
        )
        staged = dataset.process(
            config, out_dir=str(tmp_path / "staged"), progress=False,
            cache="disk", on_error="raise",
        )
        recomputed = dataset.process(
            config, out_dir=str(tmp_path / "recomputed"), progress=False,
            cache="none", on_error="raise",
        )

        assert sorted(staged.series_ids) == sorted(recomputed.series_ids)
        for series_id in staged.series_ids:
            np.testing.assert_allclose(
                staged.get(series_id).array, recomputed.get(series_id).array,
                rtol=1e-5, atol=1e-5,
            )
            np.testing.assert_array_equal(
                staged.get(series_id).mask_array, recomputed.get(series_id).mask_array
            )

    def test_staging_leaves_no_temporary_files(self, cohort_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        dataset.process(
            ProcessingConfig(segment=False, target_shape=None),
            out_dir=str(tmp_path / "out"), progress=False,
            cache="disk", cache_dir=str(cache_dir), on_error="raise",
        )
        assert list(cache_dir.iterdir()) == []

    def test_staging_records_the_whole_protocol(self, cohort_dir, tmp_path):
        """The saved config must describe the full pipeline, not just its tail."""
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        out_dir = tmp_path / "out"
        dataset.process(
            ProcessingConfig(segment=False, target_shape=None),
            out_dir=str(out_dir), progress=False, cache="disk", on_error="raise",
        )
        saved = ProcessingConfig.from_yaml(str(out_dir / "processing_config.yaml"))
        assert saved.steps == [
            "orient", "clip", "resample", "apply_mask", "standardize_size"
        ]
        assert saved.target_shape is not None

    def test_rejects_an_unknown_cache_mode(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir))
        with pytest.raises(ValueError, match="'auto', 'disk' or 'none'"):
            dataset.process(ProcessingConfig(segment=False), cache="maybe")

    def test_config_is_written_alongside_the_results(self, cohort_dir, tmp_path):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        out_dir = tmp_path / "out"
        dataset.process(
            ProcessingConfig(segment=False, target_shape=(32, 32, 8)),
            out_dir=str(out_dir), progress=False,
        )
        assert (out_dir / "processing_config.yaml").exists()
        assert (out_dir / "manifest.csv").exists()
        restored = ProcessingConfig.from_yaml(str(out_dir / "processing_config.yaml"))
        assert restored.target_shape == (32, 32, 8)

    def test_skip_existing_resumes(self, cohort_dir, tmp_path):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        config = ProcessingConfig(segment=False, target_shape=(32, 32, 8))
        out_dir = tmp_path / "out"
        dataset.process(config, out_dir=str(out_dir), progress=False)

        marker = out_dir / "case_00" / "imaging.nii.gz"
        before = marker.stat().st_mtime_ns
        dataset.process(config, out_dir=str(out_dir), progress=False, skip_existing=True)
        assert marker.stat().st_mtime_ns == before

    def test_in_memory_results_when_no_out_dir(self, cohort_dir):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        processed = dataset.process(
            ProcessingConfig(segment=False, target_shape=(24, 24, 6)), progress=False
        )
        assert all(image.shape == (24, 24, 6) for image in processed)

    def test_parallel_matches_sequential(self, cohort_dir, tmp_path):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        config = ProcessingConfig(segment=False, target_shape=(32, 32, 8))

        sequential = dataset.process(
            config, out_dir=str(tmp_path / "seq"), workers=1, progress=False
        )
        parallel = dataset.process(
            config, out_dir=str(tmp_path / "par"), workers=2, progress=False
        )
        assert sorted(sequential.series_ids) == sorted(parallel.series_ids)
        for series_id in sequential.series_ids:
            np.testing.assert_allclose(
                sequential.get(series_id).array, parallel.get(series_id).array, rtol=1e-6
            )

    def test_a_failing_case_does_not_stop_the_cohort(self, cohort_dir, tmp_path, caplog):
        dataset = Dataset.from_directory(str(cohort_dir))  # includes case_bad
        processed = dataset.process(
            ProcessingConfig(
                segment=False, dimensionality="2D", slice_selection_label=2,
                standardize_size=False,
            ),
            out_dir=str(tmp_path / "out"), progress=False, on_error="warn",
        )
        # case_bad has no mask, so 2D slice selection cannot run for it.
        assert len(processed) == 5
        assert "Failed to process case_bad" in caplog.text

    def test_on_error_raise(self, cohort_dir, tmp_path):
        dataset = Dataset.from_directory(str(cohort_dir))
        with pytest.raises(ValueError):
            dataset.process(
                ProcessingConfig(
                segment=False, dimensionality="2D", slice_selection_label=2,
                standardize_size=False,
            ),
                out_dir=str(tmp_path / "out"), progress=False, on_error="raise",
            )

    def test_flat_layout(self, cohort_dir, tmp_path):
        dataset = Dataset.from_directory(str(cohort_dir)).filter(LENIENT, progress=False)
        out_dir = tmp_path / "flat"
        dataset.process(
            ProcessingConfig(segment=False, target_shape=(32, 32, 8)),
            out_dir=str(out_dir), layout="flat", progress=False,
        )
        names = {p.name for p in out_dir.iterdir()}
        assert "case_00.nii.gz" in names
        assert "case_00_mask.nii.gz" in names
