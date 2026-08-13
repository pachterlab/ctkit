"""Tests for parsing free-text series descriptions."""

from __future__ import annotations

import pandas as pd
import pytest

from ctkit.metadata import (
    annotate,
    categorize_phase,
    categorize_region,
    summarize,
)


class TestRegion:
    @pytest.mark.parametrize(
        "description,expected",
        [
            ("CT ABDOMEN W CO", "Abdomen"),
            ("CT CHEST/ABD/PELVIS", "Chest/Abdomen/Pelvis"),
            ("CT ABDOMEN AND PELVIS", "Abdomen/Pelvis"),
            ("PET CT SKULL BASE TO MID THIGH", "Whole Body"),
            ("MRI BRAIN", "Head_neck"),
            ("CT RENAL STONE PROTOCOL", "Renal"),
            ("CT ABDOMEN RENAL", "Abdomen/Pelvis (Renal)"),
            ("something unrelated", "Other"),
        ],
    )
    def test_classification(self, description, expected):
        assert categorize_region(description) == expected

    def test_cap_shorthand(self):
        assert categorize_region("CT C/A/P") == "Chest/Abdomen/Pelvis"

    def test_missing_description(self):
        assert categorize_region(None) == "Unknown"
        assert categorize_region(float("nan")) == "Unknown"

    def test_project_vocabulary(self):
        """A collection can add its own words; TCGA prefixes are tolerated."""
        assert categorize_region("MAMMO VIEW", project="BRCA") == "Chest/Breast"
        assert categorize_region("MAMMO VIEW", project="TCGA-BRCA") == "Chest/Breast"


class TestPhase:
    @pytest.mark.parametrize(
        "description,expected",
        [
            ("Topogram 0.6 T20s", "Scout"),
            ("ARTERIAL PHASE 2.0", "Arterial"),
            ("NEPH 100 sec", "Nephrographic"),
            ("5 min delayed", "Delayed"),
            ("ABDOMEN WITHOUT CONTRAST", "Non-contrast"),
            ("POST ABDOMEN", "Post-contrast (unspecified phase)"),
            ("2.0 B31f", "Other"),
        ],
    )
    def test_classification(self, description, expected):
        assert categorize_phase(description) == expected

    def test_scout_wins_over_contrast(self):
        """Order matters: a scout is a scout even if the description says post."""
        assert categorize_phase("POST SCOUT") == "Scout"

    def test_missing_description(self):
        assert categorize_phase(None) == "Other"


class TestAnnotateAndSummarize:
    @pytest.fixture
    def metadata(self):
        return pd.DataFrame([
            {"SeriesInstanceUID": "1", "StudyInstanceUID": "s1", "PatientID": "p1",
             "Modality": "CT", "SeriesDescription": "ARTERIAL PHASE",
             "StudyDescription": "CT ABDOMEN"},
            {"SeriesInstanceUID": "2", "StudyInstanceUID": "s1", "PatientID": "p1",
             "Modality": "CT", "SeriesDescription": "Topogram",
             "StudyDescription": "CT ABDOMEN"},
            {"SeriesInstanceUID": "3", "StudyInstanceUID": "s2", "PatientID": "p2",
             "Modality": "CT", "SeriesDescription": "5 min delayed",
             "StudyDescription": "CT CHEST"},
        ])

    def test_annotate_adds_columns(self, metadata):
        annotated = annotate(metadata)
        assert list(annotated["phase"]) == ["Arterial", "Scout", "Delayed"]
        assert list(annotated["region"]) == ["Abdomen", "Abdomen", "Chest"]

    def test_annotate_needs_a_description(self):
        with pytest.raises(KeyError, match="No description column"):
            annotate(pd.DataFrame([{"PatientID": "p1"}]))

    def test_summarize_counts_distinct_entities(self, metadata):
        summary = summarize(metadata)
        totals = summary["totals"].iloc[0]
        assert totals["num_series"] == 3
        assert totals["num_studies"] == 2
        assert totals["num_patients"] == 2

    def test_summarize_breaks_down_by_phase(self, metadata):
        summary = summarize(metadata)
        assert summary["by_phase"].loc["Arterial", "num_series"] == 1
        assert summary["by_modality"].loc["CT", "num_patients"] == 2
