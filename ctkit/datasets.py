"""The catalogue of datasets this package knows how to fetch and process.

Two groups:

* the TCGA and CPTAC collections in
  :data:`~ctkit.constants.tcia_dataset_to_info`, which
  carry curated processing settings (organs to segment, clipping window,
  output dimensions);
* widely used public CT collections in :data:`EXTRA_CT_COLLECTIONS`, which are
  listed for convenience and download the same way.

Any TCIA collection can be downloaded whether or not it appears here — see
:func:`~ctkit.tcia.list_collections`, which queries
the archive directly.
"""

from __future__ import annotations

from typing import Optional

from .constants import tcia_dataset_to_info

#: Public CT collections that are commonly used as benchmarks or pretraining
#: corpora. ``clip_min,clip_max`` follows the same convention as the curated
#: TCGA/CPTAC entries: the window to clip to before feature extraction.
EXTRA_CT_COLLECTIONS = {
    "lidc-idri": {
        "collection": "LIDC-IDRI",
        "project": "tcia",
        "cancer_organ": "lung",
        "cancer_type": "lung nodules (screening)",
        "description": "1,010 thoracic CT scans with annotated lung nodules.",
        "totalsegmentator_organs": [
            "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right",
        ],
        "clip_min,clip_max": (-1000, 400),
    },
    "nsclc-radiomics": {
        "collection": "NSCLC-Radiomics",
        "project": "tcia",
        "cancer_organ": "lung",
        "cancer_type": "non-small cell lung cancer",
        "description": "422 NSCLC CT scans with manual tumor delineations (Aerts Lung1).",
        "totalsegmentator_organs": [
            "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right",
        ],
        "clip_min,clip_max": (-1000, 400),
    },
    "nsclc-radiogenomics": {
        "collection": "NSCLC Radiogenomics",
        "project": "tcia",
        "cancer_organ": "lung",
        "cancer_type": "non-small cell lung cancer",
        "description": "CT/PET with matched RNA-seq and mutation data.",
        "totalsegmentator_organs": [
            "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right",
        ],
        "clip_min,clip_max": (-1000, 400),
    },
    "pancreas-ct": {
        "collection": "Pancreas-CT",
        "project": "tcia",
        "cancer_organ": "pancreas",
        "cancer_type": "healthy pancreas (segmentation benchmark)",
        "description": "82 contrast-enhanced abdominal CT scans with pancreas masks.",
        "totalsegmentator_organs": ["pancreas"],
        "clip_min,clip_max": (-200, 300),
    },
    "hcc-tace-seg": {
        "collection": "HCC-TACE-Seg",
        "project": "tcia",
        "cancer_organ": "liver",
        "cancer_type": "hepatocellular carcinoma",
        "description": "Liver CT before/after transarterial chemoembolization, with masks.",
        "totalsegmentator_organs": ["liver"],
        "clip_min,clip_max": (-200, 400),
    },
    "colorectal-liver-metastases": {
        "collection": "Colorectal-Liver-Metastases",
        "project": "tcia",
        "cancer_organ": "liver",
        "cancer_type": "colorectal liver metastases",
        "description": "Preoperative liver CT with tumor and liver segmentations.",
        "totalsegmentator_organs": ["liver"],
        "clip_min,clip_max": (-200, 400),
    },
    "stageii-colorectal-ct": {
        "collection": "StageII-Colorectal-CT",
        "project": "tcia",
        "cancer_organ": "colon",
        "cancer_type": "stage II colorectal cancer",
        "description": "Preoperative CT for stage II colorectal cancer.",
        "totalsegmentator_organs": ["colon"],
        "clip_min,clip_max": (-200, 300),
    },
    "lungct-diagnosis": {
        "collection": "LungCT-Diagnosis",
        "project": "tcia",
        "cancer_organ": "lung",
        "cancer_type": "lung adenocarcinoma",
        "description": "Diagnostic chest CT with survival outcomes.",
        "totalsegmentator_organs": [
            "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right",
        ],
        "clip_min,clip_max": (-1000, 400),
    },
    "rider-lung-ct": {
        "collection": "RIDER Lung CT",
        "project": "tcia",
        "cancer_organ": "lung",
        "cancer_type": "non-small cell lung cancer (test-retest)",
        "description": "Same-day repeat CT scans — the standard set for testing "
                       "whether a feature is reproducible.",
        "totalsegmentator_organs": [
            "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right",
        ],
        "clip_min,clip_max": (-1000, 400),
    },
    "ct-colonography": {
        "collection": "CT COLONOGRAPHY",
        "project": "tcia",
        "cancer_organ": "colon",
        "cancer_type": "colorectal polyps (screening)",
        "description": "825 CT colonography cases; large and mostly healthy anatomy.",
        "totalsegmentator_organs": ["colon"],
        "clip_min,clip_max": (-1000, 400),
    },
}

#: Extra files that go with a collection but live outside TCIA.
SUPPLEMENTARY_DOWNLOADS = {
    "tcga-kirc": {
        "segmentations": {
            "url": "https://zenodo.org/records/13244892/files/kidney-ct.zip?download=1",
            "description": "AI and radiologist-reviewed kidney/tumor segmentations "
                           "(Scientific Reports, 2024).",
            "filename": "kidney-ct.zip",
        },
    },
}


def all_datasets() -> dict:
    """Every catalogued dataset, keyed by its short name."""
    catalogue = {}
    for key, info in tcia_dataset_to_info.items():
        entry = dict(info)
        entry.setdefault("collection", key.upper())
        entry.setdefault("curated", True)
        catalogue[key] = entry
    for key, info in EXTRA_CT_COLLECTIONS.items():
        entry = dict(info)
        entry.setdefault("curated", False)
        catalogue[key] = entry
    return catalogue


def normalize_name(name: str) -> str:
    """Accept ``TCGA-KIRC``, ``tcga_kirc`` or ``TCGA KIRC`` for the same dataset."""
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def get_dataset_info(name: str) -> dict:
    """Look up one dataset. Raises :class:`KeyError` with suggestions."""
    catalogue = all_datasets()
    key = normalize_name(name)
    if key in catalogue:
        return {"name": key, **catalogue[key]}

    # Allow the TCIA collection name itself, e.g. "NSCLC-Radiomics".
    for candidate, info in catalogue.items():
        if normalize_name(info.get("collection", candidate)) == key:
            return {"name": candidate, **info}

    close = [candidate for candidate in catalogue if key in candidate or candidate in key]
    hint = f" Did you mean: {', '.join(sorted(close))}?" if close else ""
    raise KeyError(
        f"Unknown dataset {name!r}.{hint} Use list_datasets() to see the catalogue, "
        "or pass any TCIA collection name directly to download()."
    )


def collection_for(name: str) -> str:
    """The TCIA collection name for a dataset (falls back to the name itself)."""
    try:
        return get_dataset_info(name)["collection"]
    except KeyError:
        return str(name)


def list_datasets(project: Optional[str] = None):
    """The catalogue as a DataFrame: name, collection, organ, cancer type."""
    import pandas as pd

    rows = []
    for key, info in sorted(all_datasets().items()):
        rows.append({
            "name": key,
            "collection": info.get("collection", key.upper()),
            "project": info.get("project", ""),
            "organ": info.get("cancer_organ", ""),
            "cancer_type": info.get("cancer_type", ""),
            "curated_settings": bool(info.get("curated")),
            "organs_to_segment": ", ".join(info.get("totalsegmentator_organs") or []),
            "clip_window": str(info.get("clip_min,clip_max", "")),
            "description": info.get("description", ""),
        })
    frame = pd.DataFrame(rows)
    if project:
        frame = frame[frame["project"] == project].reset_index(drop=True)
    return frame
