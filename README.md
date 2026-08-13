# ctkit

Reproducible CT image processing for AI and radiomics — from a TCIA collection
to model-ready volumes or radiomic features.

The point of this package is to make "how was this image processed?" a question
with an exact, runnable answer. A protocol is a `ProcessingConfig` object that
you can print, save to YAML, and publish alongside a paper; applying it to a
scan or to a whole cohort is one call.

![pipeline](https://github.com/pachterlab/ctkit/blob/main/figures/Fig1.png?raw=true)

```python
from ctkit import download, Dataset, ProcessingConfig

download("tcga-kirc", "data/raw", limit=20)

config = ProcessingConfig.for_dataset("tcga-kirc")
Dataset.from_directory("data/raw").filter().process(config, out_dir="data/processed")
```

Intermediate volumes only ever exist in memory. A run writes the final image,
its mask, the config that produced them, and a manifest — nothing else.

## Installation

```sh
pip install ctkit
```

Organ segmentation and radiomic features are optional extras, because they pull
in PyTorch and PyRadiomics respectively:

```sh
pip install 'ctkit[segmentation]'   # TotalSegmentator
pip install 'ctkit[radiomics]'      # PyRadiomics
pip install 'ctkit[all]'            # everything
```

Or with conda, which also brings in the non-Python tools:

```sh
conda env create -f environment.yml
conda activate ctkit
```

## The pipeline

Steps run in this order. Each is a method you can call on its own, and each is
a field in `ProcessingConfig`.

| Step | What it does | Why it matters |
| --- | --- | --- |
| `orient` | Reorient to canonical RAS | Archives disagree on storage order, so two scans of the same anatomy can arrive mirrored or transposed |
| `segment` | TotalSegmentator organ masks, merged with any tumor mask | Gives a region of interest when the collection ships without one |
| `clip` | Clamp to an intensity window (e.g. −200/300 HU) | Spends the dynamic range on the tissue you care about; caps metal artefacts |
| `resample` | Resample to a fixed voxel size in mm | Until scans share a voxel grid, a millimetre of anatomy is a different number of voxels in each one |
| `select_best_slice` | Keep the axial slice with the most tumor (2D mode) | How a 3D series becomes a 2D training example |
| `apply_mask` | Blank outside the ROI, crop to its bounding box | Removes irrelevant anatomy and makes volumes small enough to hold a cohort in memory |
| `standardize_size` | Centre-crop/pad to a common array shape | Fixed-size tensors, without rescaling the anatomy |
| `normalize` | Z-score, per volume or per dataset | Stops a model keying on per-scan intensity offsets |

Inspect a protocol before running it:

```python
>>> print(ProcessingConfig.for_dataset("tcga-kirc").describe())
Processing protocol:
  1. reorient to canonical RAS
  2. segment kidney_left, kidney_right with TotalSegmentator (task=total)
  3. clip intensities to [-200, 300] HU
  4. resample to (0.8, 0.8, 3.0) mm voxels
  5. mask to all labels, crop to ROI with 5 voxel padding
  6. crop/pad to (185, 185, 75)
```

## One image

```python
from ctkit import RadiologyImage

image = RadiologyImage("case/imaging.nii.gz", mask="case/segmentation.nii.gz")

image.orient() \
     .clip(-200, 300) \
     .resample((0.8, 0.8, 3.0)) \
     .apply_mask(labels=[1, 2], crop=True) \
     .standardize_size(185, 185, 75) \
     .save("processed/case.nii.gz")
```

Or run a whole protocol at once:

```python
image.process(ProcessingConfig.for_dataset("tcga-kirc"))
```

The constructor takes a path to a NIfTI file, a DICOM directory or zip, a
`.npy` file, or an already-loaded `Nifti1Image`, `SimpleITK.Image` or NumPy
array. Some archives distribute a "ROI" file that holds intensities inside the
region and air outside it, rather than a label map — pass `masked_roi=True` and
it is converted to a real mask on load.

Useful accessors: `.array`, `.mask_array`, `.shape`, `.spacing`, `.orientation`,
`.labels`, `.statistics()`, `.history`, and `.plot()` / `.view()` for a look at
a slice.

## A cohort

```python
from ctkit import Dataset, QCCriteria

dataset = Dataset.from_directory("data/raw")     # discovers the layout
usable = dataset.filter(QCCriteria(min_slices=25))
usable.qc_report.to_csv("qc.csv")                # every series, with reasons

processed = usable.process(config, out_dir="data/processed", workers=4)
features = processed.radiomics(labels=[1, 2], out_csv="features.csv")
```

A `Dataset` holds paths, not pixels, and loads one image at a time, so cohorts
that do not fit in memory still work.

`from_directory` recognizes three layouts: per-case directories containing
`imaging.nii.gz` (+ `segmentation.nii.gz`), a flat directory of NIfTI files
(pairing `scan.nii.gz` with `scan_mask.nii.gz`), and per-case DICOM
directories. Pass `image_pattern=`/`mask_pattern=` for anything else, or build
from a metadata table with `Dataset.from_metadata("metadata.csv")`.

### Cohort-level steps

Two steps need statistics pooled over the whole cohort: the output shape
(a percentile of the observed shapes) and dataset-level z-scoring. Leave
`target_shape=None` or set `normalization_method="dataset"` and `process()`
resolves them itself.

That needs more than one look at the data, so the pipeline splits itself: the
per-image steps — including segmentation, which dominates the runtime — run
once into a temporary directory, the statistics are measured from that, and
only the cheap crop/pad and z-scoring steps produce the final files. The
temporary directory is deleted before `process()` returns. Control it with
`cache="auto" | "disk" | "none"`.

To skip the staging entirely and run in a single pass, pin the values:

```python
config = config.replace(target_shape=(185, 185, 75))
```

## Quality control

Filtering happens at two points, because both are worth doing before you spend
time on data you will discard.

From headers alone, before downloading anything: modality, localizer/scout/MIP
keywords, slice thickness, slice count, and (for texture work) sharp
reconstruction kernels. From the reconstructed volume: 4D series, too few
slices, extreme voxel spacing, and in-plane anisotropy that marks a reformat.

```python
from ctkit import QCCriteria

QCCriteria()                      # the protocol defaults
QCCriteria.for_radiomics()        # also drops B50–B80 and bone kernels
QCCriteria.permissive()           # only drop what cannot be processed at all

result = image.check()
result.passed, result.reason, result.stats
```

## Downloading data

`download()` uses the TCIA REST API, so no Java client, manifest file or login
is needed for public collections. Series that fail the metadata checks are
skipped before their pixels are fetched.

```python
from ctkit import download, list_datasets, list_collections

list_datasets()                   # the curated catalogue
list_collections()                # every collection on TCIA, queried live

download("tcga-kirc", "data/raw", limit=20, modality="CT", workers=4)
download("NSCLC-Radiomics", "data/lung")     # any collection name works
```

The catalogue carries curated settings (organs to segment, clipping window,
output size) for the TCGA and CPTAC collections, plus commonly used public CT
collections: LIDC-IDRI, NSCLC-Radiomics, NSCLC Radiogenomics, Pancreas-CT,
HCC-TACE-Seg, Colorectal-Liver-Metastases, StageII-Colorectal-CT,
LungCT-Diagnosis, RIDER Lung CT and CT COLONOGRAPHY.

### Understanding a collection before downloading it

Series descriptions are free text (`"CT ABDOMEN W CO"`, `"ART PHASE 2.0 B31f"`).
These helpers turn them into the two fields a cohort is usually selected on —
body region and contrast phase — so you can see what a collection contains
before fetching any pixels:

```python
from ctkit import get_series, annotate, summarize

series = get_series("tcga-kirc")
summary = summarize(series, project="KIRC")
summary["by_phase"]          # how many arterial / nephrographic / delayed series

arterial = annotate(series).query("phase == 'Arterial'")
download("tcga-kirc", "data/raw", patients=arterial["PatientID"].unique())
```

The classification is heuristic — a good starting point for cohort selection,
not ground truth.

Expert segmentations that live outside TCIA are fetched separately:

```python
from ctkit import download_supplementary
download_supplementary("tcga-kirc", "data/raw", kind="segmentations")
```

Restricted collections still need the NBIA Data Retriever; see
`download_with_nbia_retriever`.

## Radiomics

```python
config = ProcessingConfig.radiomics("tcga-kirc")   # skips steps PyRadiomics repeats
features = Dataset.from_directory("data/raw").process(config).radiomics(
    labels=[1, 2], out_csv="features.csv"
)
```

`labels=[1, 2]` measures organ and tumor together; `labels=[2]` is tumor only.
Features come from the in-memory volume, so nothing intermediate is written.

## Command line

```sh
ctkit datasets                                             # what is available
ctkit download tcga-kirc --out data/raw --limit 20
ctkit filter data/raw --report qc.csv
ctkit process data/raw --out data/processed --dataset tcga-kirc
ctkit radiomics data/processed --out features.csv
ctkit info data/processed/TCGA-B0-5099/imaging.nii.gz
ctkit config --dataset tcga-kirc --out protocol.yaml       # save a protocol
```

Any step can be toggled: `--no-segment`, `--clip-min -1000`, `--spacing 1 1 1`,
`--shape 185 185 75`, `--dimensionality 2D`.

## Reproducibility

Every run writes `processing_config.yaml` next to its output. To reproduce a
dataset, or to hand one to a collaborator:

```python
config = ProcessingConfig.from_yaml("data/processed/processing_config.yaml")
Dataset.from_directory("data/raw").process(config, out_dir="rerun")
```

Each image also carries a `.history` of the steps applied to it, with the
parameters used.

## Notebooks

[`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) walks through the
package end to end: pick a collection, download it, filter it, process it, and
extract features.

## Relationship to tcia-radiology-processing

This package grew out of the protocol in
[pachterlab/tcia-radiology-processing](https://github.com/pachterlab/tcia-radiology-processing),
which documents the same pipeline as a step-by-step notebook. That repository
remains the written protocol; `ctkit` is the library implementation of it.

## License

BSD 2-Clause. See [LICENSE](LICENSE).

---

Issues and pull requests welcome.
