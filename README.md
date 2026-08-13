# ctkit

CT image processing for AI and radiomics. Makes CT processing simple and reproducible.

![pipeline](https://github.com/pachterlab/ctkit/blob/main/figures/Fig1.png?raw=true)

## Installation

```sh
pip install ctkit
```

To install pyradiomics and TotalSegmentator:

```sh
pip install 'ctkit[all]'
```

## Quick start

```python
import ctkit

ctkit.download("tcga-kirc", "data/tcga_kirc_raw", limit=20)

data = ctkit.Dataset("data/tcga_kirc_raw")
data.filter(min_slices=25).process("tcga-kirc", out_dir="data/processed")
# the tcga-kirc protocol segments the kidneys, so this one needs ctkit[all]
```

## The pipeline

| Step | What it does | Why it matters |
| --- | --- | --- |
| `filter` | Drop series that fail quality control, keeping a pass/fail table | Localizers, reformats and 4D series are not the acquisition you meant to analyze, and finding that out after processing wastes the expensive part |
| `check` | Run the same quality control without dropping anything: a result for one series, the whole table for a cohort | The measurements behind every pass and fail, which is what an exclusion criterion has to cite |
| `orient` | Reorient to canonical RAS | Archives disagree on storage order, so two scans of the same anatomy can arrive mirrored or transposed |
| `segment` | TotalSegmentator organ masks, merged with any tumor mask | Gives a region of interest when the collection ships without one |
| `clip` | Clamp to an intensity window (e.g. −200/300 HU) | Spends the dynamic range on the tissue you care about; caps metal artifacts |
| `resample` | Resample to a fixed voxel size in mm | Until scans share a voxel grid, a millimeter of anatomy is a different number of voxels in each one |
| `select_slice` | Keep one axial slice: the one with the most mask, or the one you name (2D mode) | How a 3D series becomes a 2D training example |
| `apply_mask` | Blank outside the ROI, crop to its bounding box | Removes irrelevant anatomy and makes volumes small enough to hold a cohort in memory |
| `crop_to_content` | Crop to the voxels above a threshold — air, once intensities are clipped | Trims the air around the body when there is no mask to crop to |
| `standardize_size` | Center-crop/pad to a common array shape | Fixed-size tensors, without rescaling the anatomy |
| `normalize` | Z-score, per volume or per dataset | Stops a model keying on per-scan intensity offsets |
| `save` | Write the processed series to disk | Makes the processed dataset available for training and sharing |
| `process` | Run the whole pipeline, with a saved configuration | Reproducibility and collaboration |
| `radiomics` | Extract radiomics features | For radiomic analysis. Does not require many steps above. |

## Reproducibility

Every run that writes a cohort to disk writes `processing_config.yaml` next to
it. To reproduce a dataset, or to hand one to a collaborator:

```python
ctkit.Dataset("data/raw").process(
    "data/processed/processing_config.yaml", out_dir="rerun"
)
```

`process` takes a protocol as a `ProcessingConfig`, a path to a saved one, or
the name of a collection whose curated protocol to use.

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
