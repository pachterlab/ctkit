"""Making sense of series descriptions.

TCIA metadata describes what was scanned in free text written by whoever
configured the scanner: ``"CT ABDOMEN W CO"``, ``"ART PHASE 2.0 B31f"``,
``"C/A/P"``. These helpers turn that text into the two fields a cohort is
usually selected on — which body region was imaged, and which contrast phase
the series belongs to — so a collection can be summarized and subset before
any pixels are downloaded.

The classification is heuristic. Treat it as a starting point for cohort
selection, not as ground truth.
"""

from __future__ import annotations

import re
from typing import Any, Optional

#: Body regions, matched against a normalized series or study description.
BASE_PATTERNS = {
    "chest": r"(chest|thorax|thor|lung|breast|mammo|mammary|axilla|ch\b|pa\b)",
    "abdomen": r"(abdomen|abdom|abdo|abd\b|ab\b|kub)",
    "pelvis": r"(pelvis|pelv|bladder|pel\b|hip\b)",
    "head_neck": r"(skull|head|neck|brain|c spine)",
    "whole_body": r"(pet ct|skull base to mid thigh|whole body)",
    "renal": r"(renal|kidney|kidneys|neph|ureter|urogram|uro\b|pyelo)",
}

#: Contrast phases, in priority order — the first match wins, so the specific
#: phases are tried before the generic "post-contrast".
PHASE_PATTERNS = {
    "Scout": r"(scout|topogram|surview|locator|scanogram)",
    "Non-contrast": (
        r"(non[_\-\s]?contrast|without contrast|w/o|w o\b|unenhanced|native|c-|i-|"
        r"no contrast|renal colic|stone)"
    ),
    "Arterial": r"(arterial|art\b|45 ?sec|60 ?sec|70 ?sec)",
    "Nephrographic": r"(neph|paren|90 ?sec|100 ?sec|100s|120 ?sec)",
    "Delayed": (
        r"(delay|delayed|excret|urogram|3 ?min|5 ?min|8 ?min|10 ?min|12 ?min|"
        r"15 ?min|180 ?sec)"
    ),
    "Post-contrast (unspecified phase)": r"(post|with contrast|i\+|c\+|contrast\b)",
}

#: Per-collection tweaks, for vocabulary that only makes sense in one disease.
PROJECT_OVERRIDES = {
    "BRCA": {
        "extra_patterns": {"chest": r"\b(breast|mammo|mammary|axilla)\b"},
        "rename": {"chest": "Chest/Breast"},
    },
    "KIRC": {
        "extra_patterns": {"renal": r"\b(renal|kidney|neph|ureter|urogram|stone)\b"},
        "special_rules": "renal",
    },
    "OV": {
        "extra_patterns": {
            "vascular": r"\b(vascular|aorta)\b",
            "cardiac": r"\b(cardiac)\b",
        },
    },
    "BLCA": {
        "extra_patterns": {
            "renal": r"\b(urogram|pyelo|renal|kidney|triphasic|uro)\b",
            "pelvis": r"\b(bladder)\b",
        },
    },
}

#: Typical seconds after contrast injection for each phase, used to resolve
#: series whose description does not say which phase they are.
PHASE_TIME_RANGES = {
    "Arterial": (15, 50),        # commonly 30-35 s
    "Nephrographic": (55, 100),  # commonly 65-80 s
    "Delayed": (250, 700),       # commonly 300-375 s
}


def normalize(text: str) -> str:
    """Lowercase and reduce punctuation to single spaces."""
    text = text.lower()
    text = re.sub(r"[^\w]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_tokens(text: str, *tokens: str) -> bool:
    """True when every token appears as a whole word in `text`."""
    present = text.split()
    return all(token in present for token in tokens)


def categorize_region(description: Any, project: Optional[str] = None) -> str:
    """Classify a study or series description into a body region.

    `project` selects extra vocabulary from :data:`PROJECT_OVERRIDES` (pass the
    TCGA suffix, e.g. ``"KIRC"``).
    """
    if not isinstance(description, str):
        return "Unknown"

    text = normalize(description)
    flags = {region: bool(re.search(pattern, text))
             for region, pattern in BASE_PATTERNS.items()}

    if project:
        key = str(project).upper().replace("TCGA-", "").replace("TCGA_", "")
        for region, pattern in PROJECT_OVERRIDES.get(key, {}).get("extra_patterns", {}).items():
            if re.search(pattern, text):
                flags[region] = True

    # "CAP" and "C/A/P" are shorthand for chest-abdomen-pelvis.
    if re.search(r"\bcap\b", text) or has_tokens(text, "c", "a", "p"):
        flags["chest"] = flags["abdomen"] = flags["pelvis"] = True
    elif has_tokens(text, "a", "p"):
        flags["abdomen"] = flags["pelvis"] = True
    elif has_tokens(text, "c", "a"):
        flags["chest"] = flags["abdomen"] = True

    if flags.get("whole_body"):
        return "Whole Body"

    if flags.get("renal"):
        return "Abdomen/Pelvis (Renal)" if (
            flags.get("abdomen") or flags.get("pelvis")
        ) else "Renal"

    if flags.get("chest") and flags.get("abdomen") and flags.get("pelvis"):
        return "Chest/Abdomen/Pelvis"
    if flags.get("abdomen") and flags.get("pelvis"):
        return "Abdomen/Pelvis"
    if flags.get("chest") and flags.get("abdomen"):
        return "Chest/Abdomen"

    for region in ("chest", "abdomen", "pelvis", "head_neck"):
        if flags.get(region):
            label = region.capitalize()
            if project:
                key = str(project).upper().replace("TCGA-", "").replace("TCGA_", "")
                label = PROJECT_OVERRIDES.get(key, {}).get("rename", {}).get(region, label)
            return label

    return "Other"


def categorize_phase(description: Any) -> str:
    """Classify a series description into a contrast phase."""
    if not isinstance(description, str):
        return "Other"

    text = normalize(description)
    for phase, pattern in PHASE_PATTERNS.items():
        if re.search(pattern, text):
            return phase
    return "Other"


def annotate(metadata, project: Optional[str] = None, description_column: Optional[str] = None):
    """Add ``region`` and ``phase`` columns to a series metadata table."""
    frame = metadata.copy()

    if description_column is None:
        for candidate in ("SeriesDescription", "Series Description",
                          "StudyDescription", "Study Description", "StudyDesc"):
            if candidate in frame.columns:
                description_column = candidate
                break
    if description_column is None:
        raise KeyError(
            "No description column found. Pass description_column= explicitly "
            f"(columns: {', '.join(map(str, frame.columns))})"
        )

    region_source = next(
        (column for column in ("StudyDescription", "Study Description", "StudyDesc")
         if column in frame.columns),
        description_column,
    )
    frame["region"] = frame[region_source].apply(lambda text: categorize_region(text, project))
    frame["phase"] = frame[description_column].apply(categorize_phase)
    return frame


def summarize(metadata, project: Optional[str] = None):
    """Counts of series, studies and patients, broken down several ways.

    Returns a dict of DataFrames: ``totals``, ``by_modality``,
    ``by_modality_and_region`` and ``by_phase``. This is the cohort table you
    look at before deciding what to download.
    """
    import pandas as pd

    frame = metadata if "region" in metadata.columns else annotate(metadata, project)

    series_column = _first_column(frame, "SeriesInstanceUID", "Series UID", "Series Instance UID")
    study_column = _first_column(frame, "StudyInstanceUID", "study_id", "Study UID",
                                 "Study Instance UID")
    patient_column = _first_column(frame, "PatientID", "patient_id", "Patient ID", "Subject ID")

    def counts(group_by=None):
        aggregation = {}
        if series_column:
            aggregation["num_series"] = (series_column, "nunique")
        if study_column:
            aggregation["num_studies"] = (study_column, "nunique")
        if patient_column:
            aggregation["num_patients"] = (patient_column, "nunique")
        if not aggregation:
            return pd.DataFrame({"num_rows": [len(frame)]})
        if group_by is None:
            return pd.DataFrame([{
                name: frame[column].nunique() for name, (column, _) in aggregation.items()
            }])
        return (
            frame.groupby(group_by)
            .agg(**aggregation)
            .sort_values(list(aggregation)[0], ascending=False)
        )

    summary = {"totals": counts(), "by_modality": None,
               "by_modality_and_region": None, "by_phase": counts("phase")}
    if "Modality" in frame.columns:
        summary["by_modality"] = counts("Modality")
        summary["by_modality_and_region"] = counts(["Modality", "region"])
    else:
        summary["by_modality_and_region"] = counts("region")
    return summary


def _first_column(frame, *candidates) -> Optional[str]:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None
