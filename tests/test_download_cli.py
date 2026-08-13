"""Tests for the TCIA download layer and the command line interface.

The download tests run against a stubbed archive by default so the suite does
not depend on the network. Set ``CTKIT_TEST_NETWORK=1`` to also run the tests
that contact TCIA for real.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import nibabel as nib

import pytest

from ctkit import cli, tcia as download_module


from .conftest import make_volume, network

SERIES_PAYLOAD = [
    {
        "SeriesInstanceUID": "1.2.3.100",
        "PatientID": "PATIENT-01",
        "Modality": "CT",
        "SeriesDescription": "ARTERIAL PHASE 2.0",
        "ImageCount": 120,
        "SliceThickness": 2.0,
    },
    {
        "SeriesInstanceUID": "1.2.3.101",
        "PatientID": "PATIENT-01",
        "Modality": "CT",
        "SeriesDescription": "Topogram 0.6",
        "ImageCount": 1,
    },
    {
        "SeriesInstanceUID": "1.2.3.102",
        "PatientID": "PATIENT-02",
        "Modality": "CT",
        "SeriesDescription": "PORTAL VENOUS",
        "ImageCount": 96,
        "SliceThickness": 1.5,
    },
]


class _StubResponse:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload


@pytest.fixture
def stub_archive(monkeypatch, tmp_path):
    """Replace the TCIA API with an in-process stub serving one small volume."""
    volume, _ = make_volume(shape=(16, 16, 8))
    nifti_path = tmp_path / "stub.nii.gz"
    nib.save(volume, str(nifti_path))

    def fake_get(path, params=None, timeout=None):
        if path == "getSeries":
            return _StubResponse(payload=SERIES_PAYLOAD)
        if path == "getCollectionValues":
            return _StubResponse(payload=[{"Collection": "STUB-COLLECTION"}])
        if path == "getImage":
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("LICENSE", "CC BY 3.0")
                archive.writestr("00000001.dcm", b"not really dicom")
            return _StubResponse(content=buffer.getvalue())
        raise AssertionError(f"unexpected endpoint {path}")

    def fake_dicom_to_nifti(directory, prefer="auto"):
        # The stub archive has no real DICOM in it; stand in for the converter.
        assert not os.path.exists(os.path.join(directory, "LICENSE")), (
            "the TCIA LICENSE file should be removed before conversion"
        )
        return nib.load(str(nifti_path))

    monkeypatch.setattr(download_module, "_get", fake_get)
    monkeypatch.setattr("ctkit.io.dicom_to_nifti", fake_dicom_to_nifti)
    return nifti_path


class TestDownload:
    def test_downloads_and_converts(self, stub_archive, tmp_path):
        out_dir = str(tmp_path / "raw")
        dataset = download_module.download("stub-collection", out_dir, progress=False, workers=1)

        assert len(dataset) == 2, "the topogram should have been filtered out"
        assert os.path.exists(os.path.join(out_dir, "PATIENT-01", "imaging.nii.gz"))
        assert os.path.exists(os.path.join(out_dir, "metadata.csv"))
        assert os.path.exists(os.path.join(out_dir, "excluded_series.csv"))

    def test_excluded_series_are_recorded_with_reasons(self, stub_archive, tmp_path):
        import pandas as pd

        out_dir = str(tmp_path / "raw")
        download_module.download("stub-collection", out_dir, progress=False, workers=1)
        excluded = pd.read_csv(os.path.join(out_dir, "excluded_series.csv"))
        assert len(excluded) == 1
        assert "topogram" in excluded.iloc[0]["exclusion_reason"]

    def test_no_filter_keeps_everything(self, stub_archive, tmp_path):
        dataset = download_module.download(
            "stub-collection", str(tmp_path / "raw"),
            filter_series=False, progress=False, workers=1,
        )
        assert len(dataset) == 3

    def test_limit(self, stub_archive, tmp_path):
        dataset = download_module.download(
            "stub-collection", str(tmp_path / "raw"), limit=1, progress=False, workers=1
        )
        assert len(dataset) == 1

    def test_series_ids_are_unique_per_patient(self, stub_archive, tmp_path):
        dataset = download_module.download(
            "stub-collection", str(tmp_path / "raw"),
            filter_series=False, progress=False, workers=1,
        )
        assert sorted(dataset.series_ids) == ["PATIENT-01", "PATIENT-01_2", "PATIENT-02"]

    def test_metadata_travels_with_the_images(self, stub_archive, tmp_path):
        dataset = download_module.download(
            "stub-collection", str(tmp_path / "raw"), progress=False, workers=1
        )
        assert dataset.get("PATIENT-01").metadata["SeriesDescription"] == "ARTERIAL PHASE 2.0"

    def test_resumes_without_redownloading(self, stub_archive, tmp_path):
        out_dir = str(tmp_path / "raw")
        download_module.download("stub-collection", out_dir, progress=False, workers=1)
        marker = os.path.join(out_dir, "PATIENT-01", "imaging.nii.gz")
        before = os.stat(marker).st_mtime_ns
        download_module.download("stub-collection", out_dir, progress=False, workers=1)
        assert os.stat(marker).st_mtime_ns == before

    def test_get_series_returns_a_frame(self, stub_archive):
        frame = download_module.get_series("stub-collection")
        assert len(frame) == 3
        assert "SeriesInstanceUID" in frame.columns

    def test_unknown_supplementary_download(self):
        with pytest.raises(KeyError, match="No 'segmentations' download registered"):
            download_module.download_supplementary("tcga-luad", "/tmp/x")

    def test_nbia_retriever_missing_is_explained(self, tmp_path):
        manifest = tmp_path / "m.tcia"
        manifest.write_text("")
        with pytest.raises(FileNotFoundError, match="NBIA Data Retriever|not on the PATH"):
            download_module.download_with_nbia_retriever(
                str(manifest), str(tmp_path), executable="definitely-not-installed"
            )


@network
class TestDownloadAgainstTCIA:
    def test_lists_collections(self):
        collections = download_module.list_collections()
        assert "TCGA-KIRC" in collections
        assert len(collections) > 50

    def test_series_metadata_for_a_real_collection(self):
        frame = download_module.get_series("tcga-kich", modality="CT")
        assert len(frame) > 10
        assert {"SeriesInstanceUID", "PatientID", "SeriesDescription"} <= set(frame.columns)

    def test_downloads_one_real_series(self, tmp_path):
        dataset = download_module.download(
            "tcga-kich", str(tmp_path / "raw"), limit=1, progress=False, workers=1
        )
        assert len(dataset) == 1
        image = dataset[0]
        assert image.ndim == 3 and image.shape[2] > 10


class TestCLI:
    def test_help_exits_cleanly(self, capsys):
        with pytest.raises(SystemExit) as caught:
            cli.main(["--help"])
        assert caught.value.code == 0
        assert "reproducible" in capsys.readouterr().out.lower()

    def test_no_command_prints_usage(self, capsys):
        assert cli.main([]) == 1
        assert "usage" in capsys.readouterr().out.lower()

    def test_datasets_command(self, capsys):
        assert cli.main(["datasets"]) == 0
        output = capsys.readouterr().out
        assert "tcga-kirc" in output and "lidc-idri" in output

    def test_config_command_emits_yaml(self, capsys):
        assert cli.main(["config", "--dataset", "tcga-kirc"]) == 0
        output = capsys.readouterr().out
        assert "clip_min: -200" in output
        assert "kidney_left" in output

    def test_config_command_writes_a_file(self, tmp_path, capsys):
        path = str(tmp_path / "config.yaml")
        assert cli.main(["config", "--preset", "radiomics", "--out", path]) == 0
        from ctkit import ProcessingConfig

        assert ProcessingConfig.from_yaml(path).resample is False

    def test_info_command(self, case_dir, capsys):
        assert cli.main(["info", str(case_dir / "imaging.nii.gz")]) == 0
        output = capsys.readouterr().out
        assert "orientation" in output and "quality_control" in output

    def test_info_command_json(self, case_dir, capsys):
        assert cli.main(["info", str(case_dir / "imaging.nii.gz"), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["series_id"] == "case_00"

    def test_filter_command_reports(self, cohort_dir, tmp_path, capsys):
        report = str(tmp_path / "qc.csv")
        assert cli.main([
            "filter", str(cohort_dir), "--report", report, "--min-slices", "10"
        ]) == 0
        assert os.path.exists(report)
        assert "case_bad" in capsys.readouterr().out

    def test_process_command(self, cohort_dir, tmp_path, capsys):
        out_dir = str(tmp_path / "out")
        assert cli.main([
            "--log-level", "ERROR",
            "process", str(cohort_dir), "--out", out_dir,
            "--no-segment", "--shape", "32", "32", "8",
        ]) == 0
        assert os.path.exists(os.path.join(out_dir, "case_00", "imaging.nii.gz"))
        assert "Processed" in capsys.readouterr().out

    def test_process_command_single_file(self, case_dir, tmp_path, capsys):
        out_path = str(tmp_path / "one.nii.gz")
        assert cli.main([
            "--log-level", "ERROR",
            "process", str(case_dir / "imaging.nii.gz"), "--out", out_path,
            "--no-segment", "--no-mask", "--no-standardize-size",
        ]) == 0
        assert os.path.exists(out_path)

    def test_step_flags_override_the_config(self, cohort_dir, tmp_path):
        out_dir = str(tmp_path / "out")
        cli.main([
            "--log-level", "ERROR",
            "process", str(cohort_dir), "--out", out_dir,
            "--no-segment", "--no-clip", "--no-resample", "--no-mask",
            "--no-standardize-size",
        ])
        from ctkit import ProcessingConfig

        written = ProcessingConfig.from_yaml(os.path.join(out_dir, "processing_config.yaml"))
        assert written.steps == ["orient"]

    def test_segmentation_dir_reaches_the_config(self, tmp_path):
        args = cli._build_parser().parse_args([
            "process", "in", "--out", "out",
            "--organs", "kidney_left",
            "--segmentation-dir", str(tmp_path / "seg"),
        ])
        config = cli._config_from_args(args)
        assert config.segmentation_dir == str(tmp_path / "seg")
        assert config.segment  # asking to keep the masks enables the step

    def test_segmentation_dir_does_not_re_enable_a_skipped_step(self, tmp_path):
        args = cli._build_parser().parse_args([
            "process", "in", "--out", "out",
            "--no-segment", "--segmentation-dir", str(tmp_path / "seg"),
        ])
        assert cli._config_from_args(args).segment is False

    def test_errors_are_reported_without_a_traceback(self, capsys):
        assert cli.main(["process", "/nonexistent", "--out", "/tmp/x"]) == 1
        assert "error:" in capsys.readouterr().err


@pytest.fixture
def metadata_csv(tmp_path):
    """A metadata table with one diagnostic series per unusable one."""
    import pandas as pd

    frame = pd.DataFrame([
        {"series_id": "case_00", "Modality": "CT",
         "SeriesDescription": "ARTERIAL PHASE 2.0", "SliceThickness": 2.0,
         "ImageCount": 120},
        {"series_id": "case_01", "Modality": "CT",
         "SeriesDescription": "CT ABDOMEN B70f SHARP", "SliceThickness": 1.5,
         "ImageCount": 90},
        {"series_id": "case_02", "Modality": "CT",
         "SeriesDescription": "SCOUT", "SliceThickness": 1.0, "ImageCount": 2},
        {"series_id": "case_03", "Modality": "MR",
         "SeriesDescription": "T2 AXIAL", "SliceThickness": 3.0, "ImageCount": 40},
        {"series_id": "case_bad", "Modality": "CT",
         "SeriesDescription": "CT ABDOMEN", "SliceThickness": 12.0,
         "ImageCount": 60},
    ])
    path = tmp_path / "metadata.csv"
    frame.to_csv(path, index=False)
    return path


class TestFilterCommand:
    """The filter command, at both levels the protocol filters at."""

    def _read(self, path):
        import pandas as pd

        return pd.read_csv(path)

    def test_directory_reports_and_excludes(self, cohort_dir, tmp_path, capsys):
        report = str(tmp_path / "qc.csv")
        assert cli.main([
            "filter", str(cohort_dir), "--report", report, "--min-slices", "10"
        ]) == 0
        assert "case_bad" in capsys.readouterr().out
        frame = self._read(report)
        assert not frame.set_index("series_id").loc["case_bad", "passed"]

    def test_metadata_csv_needs_no_pixels(self, metadata_csv, tmp_path, capsys):
        out = str(tmp_path / "kept.csv")
        assert cli.main([
            "filter", str(metadata_csv), "--metadata-out", out
        ]) == 0

        kept = self._read(out)
        assert list(kept["series_id"]) == ["case_00", "case_01"]  # scout, MR, thick gone
        assert kept["qc_passed"].all()
        assert "3 of 5 series passed" not in capsys.readouterr().out  # 2 of 5

    def test_metadata_csv_keeps_rejected_rows_when_asked(self, metadata_csv, tmp_path):
        out = str(tmp_path / "annotated.csv")
        cli.main([
            "filter", str(metadata_csv), "--metadata-out", out, "--keep-rejected-rows"
        ])
        frame = self._read(out).set_index("series_id")
        assert len(frame) == 5
        assert "scout" in frame.loc["case_02", "qc_reason"].lower()

    def test_radiomics_preset_drops_sharp_kernels(self, metadata_csv, tmp_path):
        out = str(tmp_path / "kept.csv")
        cli.main([
            "filter", str(metadata_csv), "--preset", "radiomics", "--metadata-out", out
        ])
        assert list(self._read(out)["series_id"]) == ["case_00"]

    def test_criteria_flags_override_the_preset(self, metadata_csv, tmp_path):
        out = str(tmp_path / "kept.csv")
        cli.main([
            "filter", str(metadata_csv), "--modality", "any",
            "--max-slice-thickness", "20", "--metadata-out", out,
        ])
        kept = list(self._read(out)["series_id"])
        assert "case_03" in kept and "case_bad" in kept  # MR and thick slices allowed
        assert "case_02" not in kept                      # the scout still goes

    def test_metadata_level_skips_the_volume_checks(self, cohort_dir, tmp_path):
        report = str(tmp_path / "qc.csv")
        assert cli.main([
            "filter", str(cohort_dir), "--level", "metadata", "--report", report
        ]) == 0
        frame = self._read(report)
        assert frame["passed"].all()  # nothing in the headers disqualifies these

    def test_annotates_an_existing_metadata_table(
        self, cohort_dir, metadata_csv, tmp_path
    ):
        table = self._read(metadata_csv)
        table.loc[len(table)] = {**table.iloc[0].to_dict(), "series_id": "case_elsewhere"}
        table.to_csv(metadata_csv, index=False)

        out = str(tmp_path / "metadata_filtered.csv")
        assert cli.main([
            "filter", str(cohort_dir), "--min-slices", "10",
            "--metadata", str(metadata_csv), "--metadata-out", out,
        ]) == 0
        frame = self._read(out)
        assert "case_bad" not in list(frame["series_id"])        # 3 slices: dropped
        assert "case_elsewhere" in list(frame["series_id"])      # not measured: kept
        assert frame.set_index("series_id").loc["case_00", "qc_passed"]

    def test_moves_rejected_series_aside(self, cohort_dir, tmp_path, capsys):
        excluded = str(tmp_path / "excluded")
        assert cli.main([
            "filter", str(cohort_dir), "--min-slices", "10", "--rejected-out", excluded
        ]) == 0
        assert os.path.exists(os.path.join(excluded, "case_bad", "imaging.nii.gz"))
        assert not os.path.exists(os.path.join(str(cohort_dir), "case_bad"))
        assert os.path.exists(os.path.join(str(cohort_dir), "case_00"))

    def test_deleting_asks_first(self, cohort_dir, capsys):
        assert cli.main([
            "filter", str(cohort_dir), "--min-slices", "10", "--delete-rejected"
        ]) == 0
        assert os.path.exists(os.path.join(str(cohort_dir), "case_bad"))
        assert "--yes" in capsys.readouterr().err

    def test_deleting_with_yes(self, cohort_dir):
        assert cli.main([
            "filter", str(cohort_dir), "--min-slices", "10",
            "--delete-rejected", "--yes",
        ]) == 0
        assert not os.path.exists(os.path.join(str(cohort_dir), "case_bad"))
        assert os.path.exists(os.path.join(str(cohort_dir), "case_00"))

    def test_volume_level_needs_images(self, metadata_csv, capsys):
        assert cli.main(["filter", str(metadata_csv), "--level", "volume"]) == 1
        assert "needs images" in capsys.readouterr().err
