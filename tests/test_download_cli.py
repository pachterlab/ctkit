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

    def test_errors_are_reported_without_a_traceback(self, capsys):
        assert cli.main(["process", "/nonexistent", "--out", "/tmp/x"]) == 1
        assert "error:" in capsys.readouterr().err
