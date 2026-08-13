"""Downloading imaging collections from The Cancer Imaging Archive.

Uses the public NBIA REST API, so no Java client or manifest file is needed:
series metadata comes from ``getSeries`` and pixel data from ``getImage``,
which returns a zip of DICOM slices.

Each series is converted to NIfTI in a temporary directory and only the
converted volume is kept, so a collection lands on disk as

    out_dir/<series_id>/imaging.nii.gz

Collections with restricted access still need the NBIA Data Retriever and a
signed license; :func:`download_with_nbia_retriever` wraps that path.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from typing import Optional, Sequence

from .datasets import SUPPLEMENTARY_DOWNLOADS, collection_for, get_dataset_info, normalize_name
from .qc import QCCriteria, check_series_metadata

logger = logging.getLogger(__name__)

TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 3


class TCIAError(RuntimeError):
    """A request to the TCIA archive failed."""


# ----------------------------------------------------------------------
# API access
# ----------------------------------------------------------------------
def _session():
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "ctkit"})
    return session


def _get(path: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT):
    import requests

    url = f"{TCIA_API}/{path}"
    last_error: Optional[Exception] = None
    for attempt in range(DEFAULT_RETRIES):
        try:
            response = _session().get(url, params=params or {}, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            wait = 2 ** attempt
            logger.warning(
                "TCIA request failed (%s); retrying in %ds [%d/%d]",
                error, wait, attempt + 1, DEFAULT_RETRIES,
            )
            time.sleep(wait)
    raise TCIAError(f"GET {url} failed after {DEFAULT_RETRIES} attempts: {last_error}")


def list_collections() -> list:
    """Every collection name available from TCIA, queried live."""
    payload = _get("getCollectionValues").json()
    return sorted(entry["Collection"] for entry in payload)


def get_series(
    collection: str,
    modality: Optional[str] = "CT",
    patient_id: Optional[str] = None,
    body_part: Optional[str] = None,
):
    """Series-level metadata for a collection, as a DataFrame.

    One row per imaging series, with description, slice count, manufacturer and
    the UIDs needed to fetch the pixels. This is enough to decide what to
    download before downloading anything.
    """
    import pandas as pd

    params: dict = {"Collection": collection_for(collection)}
    if modality:
        params["Modality"] = modality
    if patient_id:
        params["PatientID"] = patient_id
    if body_part:
        params["BodyPartExamined"] = body_part

    payload = _get("getSeries", params).json()
    frame = pd.DataFrame(payload)
    if frame.empty:
        raise TCIAError(
            f"TCIA returned no series for collection {params['Collection']!r}"
            + (f" with modality {modality!r}" if modality else "")
            + ". Check the name with list_collections()."
        )
    return frame


def download_series(
    series_uid: str,
    out_dir: str,
    convert: bool = True,
    series_id: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[str]:
    """Download one series and write it as NIfTI (or raw DICOM).

    Returns the path written, or ``None`` if it already existed.
    """
    series_id = series_id or series_uid
    case_dir = os.path.join(out_dir, series_id)
    target = os.path.join(case_dir, "imaging.nii.gz" if convert else "")

    if convert and os.path.exists(target) and not overwrite:
        logger.debug("%s already downloaded", series_id)
        return None
    if not convert and os.path.isdir(case_dir) and os.listdir(case_dir) and not overwrite:
        logger.debug("%s already downloaded", series_id)
        return None

    with tempfile.TemporaryDirectory(prefix="ctkit_dl_") as tmp:
        archive_path = os.path.join(tmp, "series.zip")
        response = _get("getImage", {"SeriesInstanceUID": series_uid})
        with open(archive_path, "wb") as handle:
            handle.write(response.content)

        extracted = os.path.join(tmp, "dicom")
        os.makedirs(extracted, exist_ok=True)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extracted)
        except zipfile.BadZipFile as error:
            raise TCIAError(
                f"TCIA returned a corrupt archive for series {series_uid}: {error}"
            ) from error

        # TCIA ships a LICENSE file alongside the slices.
        license_path = os.path.join(extracted, "LICENSE")
        if os.path.exists(license_path):
            os.remove(license_path)

        os.makedirs(case_dir, exist_ok=True)
        if not convert:
            for name in os.listdir(extracted):
                shutil.move(os.path.join(extracted, name), os.path.join(case_dir, name))
            return case_dir

        from .io import dicom_to_nifti, save_image

        image = dicom_to_nifti(extracted)
        return save_image(image, target)


def download(
    dataset: str,
    out_dir: str,
    modality: Optional[str] = "CT",
    limit: Optional[int] = None,
    patients: Optional[Sequence[str]] = None,
    convert: bool = True,
    filter_series: bool = True,
    criteria: Optional[QCCriteria] = None,
    overwrite: bool = False,
    progress: bool = True,
    workers: int = 4,
):
    """Download a collection and return it as a :class:`~.dataset.Dataset`.

    Parameters
    ----------
    dataset:
        A catalog name (``"tcga-kirc"``) or any TCIA collection name
        (``"NSCLC-Radiomics"``).
    out_dir:
        Destination. Each series becomes ``out_dir/<series_id>/imaging.nii.gz``.
    modality:
        Restrict to one modality. ``None`` downloads every modality.
    limit:
        Stop after this many series — useful for a trial run.
    patients:
        Restrict to these ``PatientID`` values.
    convert:
        Convert each series to NIfTI (the default). ``False`` keeps the DICOM
        slices instead.
    filter_series:
        Drop localizers, scouts, MIPs and thick-slice series using the metadata
        TCIA returns, *before* downloading them.
    workers:
        Parallel downloads. The archive is the bottleneck, not the CPU.

    Returns
    -------
    Dataset
        The downloaded cohort. ``metadata.csv`` and, when filtering,
        ``excluded_series.csv`` are written alongside it.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .dataset import Dataset, _progress

    collection = collection_for(dataset)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Querying TCIA for collection %r", collection)
    series = get_series(collection, modality=modality)
    logger.info("%d series found in %s", len(series), collection)

    if patients:
        wanted = set(patients)
        series = series[series["PatientID"].isin(wanted)]
        if series.empty:
            raise TCIAError(f"No series in {collection} for patients {sorted(wanted)}")

    excluded = None
    if filter_series:
        series, excluded = _filter_series_metadata(series, criteria)
        logger.info(
            "%d series pass metadata filtering (%d excluded before download)",
            len(series), 0 if excluded is None else len(excluded),
        )
        if excluded is not None and len(excluded):
            excluded.to_csv(os.path.join(out_dir, "excluded_series.csv"), index=False)

    series = series.sort_values(["PatientID", "SeriesInstanceUID"]).reset_index(drop=True)
    series["series_id"] = _make_series_ids(series)
    if limit is not None:
        series = series.head(limit).copy()

    try:
        info = get_dataset_info(dataset)
        series["dataset"] = info["name"]
        series["cancer_organ"] = info.get("cancer_organ", "")
        series["cancer_type"] = info.get("cancer_type", "")
    except KeyError:
        series["dataset"] = normalize_name(dataset)

    logger.info("Downloading %d series to %s", len(series), out_dir)
    failures = []

    def fetch(row) -> Optional[str]:
        return download_series(
            row["SeriesInstanceUID"],
            out_dir,
            convert=convert,
            series_id=row["series_id"],
            overwrite=overwrite,
        )

    rows = [row for _, row in series.iterrows()]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, row): row for row in rows}
            for future in _progress(
                as_completed(futures), f"Downloading {collection}", progress, total=len(futures)
            ):
                row = futures[future]
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - one bad series is not fatal
                    logger.error("Failed to download %s: %s", row["series_id"], error)
                    failures.append({"series_id": row["series_id"], "error": str(error)})
    else:
        for row in _progress(rows, f"Downloading {collection}", progress):
            try:
                fetch(row)
            except Exception as error:  # noqa: BLE001
                logger.error("Failed to download %s: %s", row["series_id"], error)
                failures.append({"series_id": row["series_id"], "error": str(error)})

    if failures:
        import pandas as pd

        pd.DataFrame(failures).to_csv(os.path.join(out_dir, "failed_series.csv"), index=False)
        logger.warning(
            "%d of %d series failed to download; see failed_series.csv",
            len(failures), len(series),
        )

    metadata_path = os.path.join(out_dir, "metadata.csv")
    series.to_csv(metadata_path, index=False)
    logger.info("Wrote series metadata to %s", metadata_path)

    downloaded = Dataset.from_directory(out_dir, name=normalize_name(dataset))
    _attach_metadata(downloaded, series)
    return downloaded


def download_supplementary(dataset: str, out_dir: str, kind: str = "segmentations") -> str:
    """Fetch extra files that go with a collection, such as expert segmentations.

    See :data:`~ctkit.datasets.SUPPLEMENTARY_DOWNLOADS` for
    what is available.
    """
    key = normalize_name(dataset)
    available = SUPPLEMENTARY_DOWNLOADS.get(key, {})
    if kind not in available:
        raise KeyError(
            f"No {kind!r} download registered for {dataset!r}. "
            f"Available for this dataset: {', '.join(available) or 'nothing'}."
        )

    entry = available[kind]
    os.makedirs(out_dir, exist_ok=True)
    archive_path = os.path.join(out_dir, entry["filename"])

    if not os.path.exists(archive_path):
        logger.info("Downloading %s: %s", kind, entry["description"])
        _download_url(entry["url"], archive_path)

    if archive_path.endswith(".zip"):
        target = os.path.join(out_dir, os.path.splitext(entry["filename"])[0])
        if not os.path.isdir(target):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(target)
            logger.info("Extracted to %s", target)
        return target
    return archive_path


def _download_url(url: str, path: str, chunk_size: int = 1 << 20) -> str:
    import requests

    with requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=total or None, unit="B", unit_scale=True,
                       desc=os.path.basename(path))
        except ImportError:  # pragma: no cover
            bar = None

        with open(path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                handle.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if bar is not None:
            bar.close()
    return path


def download_with_nbia_retriever(
    manifest: str,
    out_dir: str,
    executable: str = "nbia-data-retriever",
    accept_license: bool = False,
) -> str:
    """Download from a ``.tcia`` manifest using the NBIA Data Retriever.

    Needed only for collections with restricted access, where the REST API
    will not serve pixel data without credentials.
    """
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"{executable!r} is not on the PATH. Install the NBIA Data Retriever from "
            "https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images "
            "or use download(), which does not need it for public collections."
        )
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    os.makedirs(out_dir, exist_ok=True)
    command = f"{executable} --cli {manifest} -d {out_dir} -v -f"
    if accept_license:
        command = "yes 'Y\nM' | " + command
    logger.info("Running: %s", command)
    subprocess.run(command, shell=True, check=True)
    return out_dir


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _filter_series_metadata(series, criteria: Optional[QCCriteria]):
    """Split a series table into kept and excluded, using headers alone."""
    criteria = criteria or QCCriteria()
    keep_rows, drop_rows = [], []

    for _, row in series.iterrows():
        result = check_series_metadata(row.to_dict(), criteria)
        if result.passed:
            keep_rows.append(row)
        else:
            record = row.to_dict()
            record["exclusion_reason"] = result.reason
            drop_rows.append(record)

    import pandas as pd

    kept = pd.DataFrame(keep_rows).reset_index(drop=True) if keep_rows else series.iloc[0:0]
    excluded = pd.DataFrame(drop_rows) if drop_rows else None
    return kept, excluded


def _make_series_ids(series) -> list:
    """Readable, stable, unique ids: the patient id, suffixed when repeated."""
    counts: dict = {}
    ids = []
    for patient_id in series["PatientID"].astype(str):
        counts[patient_id] = counts.get(patient_id, 0) + 1
        index = counts[patient_id]
        ids.append(patient_id if index == 1 else f"{patient_id}_{index}")
    return ids


def _attach_metadata(dataset, series) -> None:
    """Copy each metadata row onto the matching image in `dataset`."""
    by_id = {str(row["series_id"]): row.to_dict() for _, row in series.iterrows()}
    for image in dataset:
        record = by_id.get(str(image.series_id))
        if record:
            image.metadata.update(record)
