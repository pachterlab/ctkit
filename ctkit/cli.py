"""Command line interface: ``ctkit <command>``.

Every command mirrors a piece of the Python API, so a protocol can be run from
a shell script or a workflow manager without writing Python.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional, Sequence

from . import __version__


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    try:
        return args.handler(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - the CLI reports, it does not trace
        if args.log_level.upper() == "DEBUG":
            raise
        print(f"error: {error}", file=sys.stderr)
        return 1


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctkit",
        description="Reproducible radiology image processing for AI and radiomics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  ctkit datasets\n"
            "  ctkit download tcga-kirc --out data/raw --limit 20\n"
            "  ctkit process data/raw --out data/processed --dataset tcga-kirc\n"
            "  ctkit filter data/raw --report qc.csv\n"
            "  ctkit radiomics data/processed --out features.csv\n"
            "  ctkit info data/processed/TCGA-KN-8424/imaging.nii.gz\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="verbosity (default: INFO)",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_datasets_command(subparsers)
    _add_download_command(subparsers)
    _add_process_command(subparsers)
    _add_filter_command(subparsers)
    _add_radiomics_command(subparsers)
    _add_info_command(subparsers)
    _add_config_command(subparsers)
    return parser


def _add_datasets_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "datasets", help="list the datasets this package knows about"
    )
    parser.add_argument("--project", help="restrict to a project, e.g. tcga or cptac")
    parser.add_argument(
        "--tcia", action="store_true",
        help="list every collection available from TCIA (queries the archive)",
    )
    parser.set_defaults(handler=_run_datasets)


def _add_download_command(subparsers) -> None:
    parser = subparsers.add_parser("download", help="download a collection from TCIA")
    parser.add_argument("dataset", help="catalogue name or TCIA collection name")
    parser.add_argument("--out", required=True, help="destination directory")
    parser.add_argument("--modality", default="CT", help="modality filter (default: CT)")
    parser.add_argument("--limit", type=int, help="stop after this many series")
    parser.add_argument("--patients", nargs="+", help="restrict to these PatientIDs")
    parser.add_argument(
        "--no-convert", action="store_true",
        help="keep DICOM slices instead of converting to NIfTI",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="download every series, including localizers and scouts",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--supplementary", metavar="KIND",
        help="also fetch supplementary files, e.g. 'segmentations'",
    )
    parser.set_defaults(handler=_run_download)


def _add_process_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "process", help="run the full processing pipeline over a directory of images"
    )
    parser.add_argument("input", help="directory of images (or a single image file)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--config", help="a processing config YAML (see `ctkit config`)")
    parser.add_argument("--dataset", help="use the curated settings for this collection")
    parser.add_argument(
        "--dimensionality", choices=["2D", "3D"], help="3D volumes or the best 2D slice"
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument(
        "--layout", choices=["case_dirs", "flat"], default="case_dirs",
        help="output layout (default: case_dirs)",
    )
    parser.add_argument(
        "--format", dest="output_format", choices=["nifti", "numpy"], default="nifti"
    )
    parser.add_argument("--skip-existing", action="store_true", help="resume a partial run")
    parser.add_argument("--qc", action="store_true", help="run quality control first")
    parser.add_argument(
        "--image-pattern", help="glob for image files, e.g. '*_CT.nii.gz'"
    )
    parser.add_argument("--mask-pattern", help="glob for mask files")

    steps = parser.add_argument_group("pipeline steps (override the config)")
    for name, help_text in [
        ("orient", "reorient to canonical RAS"),
        ("segment", "segment organs with TotalSegmentator"),
        ("clip", "clip the intensity window"),
        ("resample", "resample to a common voxel size"),
        ("mask", "mask and crop to the region of interest"),
        ("standardize-size", "crop/pad to a common array shape"),
        ("normalize", "z-score the intensities"),
    ]:
        steps.add_argument(f"--{name}", dest=name.replace("-", "_"),
                           action="store_true", default=None, help=help_text)
        steps.add_argument(f"--no-{name}", dest=name.replace("-", "_"),
                           action="store_false", default=None,
                           help=f"skip: {help_text}")

    steps.add_argument("--clip-min", type=float, help="lower intensity bound (HU)")
    steps.add_argument("--clip-max", type=float, help="upper intensity bound (HU)")
    steps.add_argument(
        "--spacing", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="target voxel spacing in mm",
    )
    steps.add_argument(
        "--shape", nargs=3, type=int, metavar=("X", "Y", "Z"),
        help="target array shape (default: the 95th percentile of the cohort)",
    )
    steps.add_argument("--organs", nargs="+", help="TotalSegmentator structures to segment")
    parser.set_defaults(handler=_run_process)


def _add_filter_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "filter", help="run quality control and report which series are usable"
    )
    parser.add_argument("input", help="directory of images")
    parser.add_argument("--report", help="write the full pass/fail table to this CSV")
    parser.add_argument("--out", help="copy the series that pass into this directory")
    parser.add_argument("--min-slices", type=int, default=25)
    parser.add_argument("--max-spacing", type=float, default=20.0)
    parser.add_argument("--max-anisotropy", type=float, default=4.0)
    parser.add_argument("--allow-4d", action="store_true")
    parser.set_defaults(handler=_run_filter)


def _add_radiomics_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "radiomics", help="extract PyRadiomics features from processed images"
    )
    parser.add_argument("input", help="directory of processed images with masks")
    parser.add_argument("--out", required=True, help="output CSV")
    parser.add_argument(
        "--labels", nargs="+", type=int, default=[1, 2],
        help="mask labels making up the region of interest (default: 1 2)",
    )
    parser.add_argument("--params", help="a PyRadiomics parameter YAML")
    parser.set_defaults(handler=_run_radiomics)


def _add_info_command(subparsers) -> None:
    parser = subparsers.add_parser("info", help="describe an image or a directory of images")
    parser.add_argument("input")
    parser.add_argument("--mask", help="a mask to describe alongside it")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.set_defaults(handler=_run_info)


def _add_config_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "config", help="print or save a processing config"
    )
    parser.add_argument("--dataset", help="start from the curated settings for a collection")
    parser.add_argument("--preset", choices=["default", "radiomics", "minimal"],
                        default="default")
    parser.add_argument("--out", help="write the YAML here instead of printing it")
    parser.set_defaults(handler=_run_config)


# ----------------------------------------------------------------------
# handlers
# ----------------------------------------------------------------------
def _run_datasets(args) -> int:
    if args.tcia:
        from .tcia import list_collections

        for name in list_collections():
            print(name)
        return 0

    from .datasets import list_datasets

    frame = list_datasets(project=args.project)
    columns = ["name", "collection", "organ", "cancer_type", "curated_settings"]
    with _wide_output():
        print(frame[columns].to_string(index=False))
    print(
        f"\n{len(frame)} datasets. Curated entries carry a clipping window, organ list "
        "and output size; the rest use protocol defaults.\n"
        "Any TCIA collection also works: `ctkit datasets --tcia` lists them all."
    )
    return 0


def _run_download(args) -> int:
    from .tcia import download, download_supplementary

    dataset = download(
        args.dataset,
        args.out,
        modality=args.modality,
        limit=args.limit,
        patients=args.patients,
        convert=not args.no_convert,
        filter_series=not args.no_filter,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    print(f"Downloaded {len(dataset)} series to {args.out}")

    if args.supplementary:
        path = download_supplementary(args.dataset, args.out, kind=args.supplementary)
        print(f"Supplementary {args.supplementary} in {path}")
    return 0


def _run_process(args) -> int:
    from .dataset import Dataset
    from .image import RadiologyImage

    config = _config_from_args(args)

    if os.path.isfile(args.input):
        image = RadiologyImage(args.input).process(config)
        written = image.save(
            args.out, output_format=config.output_format, compress=config.compress
        )
        print(f"Wrote {written}")
        return 0

    dataset = Dataset.from_directory(
        args.input,
        image_pattern=args.image_pattern,
        mask_pattern=args.mask_pattern,
    )
    print(f"Found {len(dataset)} series in {args.input}")
    print(config.describe())

    if args.qc:
        dataset = dataset.filter()

    processed = dataset.process(
        config,
        out_dir=args.out,
        workers=args.workers,
        layout=args.layout,
        skip_existing=args.skip_existing,
    )
    print(f"Processed {len(processed)} of {len(dataset)} series into {args.out}")
    return 0


def _run_filter(args) -> int:
    from .dataset import Dataset
    from .qc import QCCriteria

    criteria = QCCriteria(
        min_slices=args.min_slices,
        max_spacing=args.max_spacing,
        max_in_plane_anisotropy=args.max_anisotropy,
        reject_4d=not args.allow_4d,
    )
    dataset = Dataset.from_directory(args.input)
    kept = dataset.filter(criteria)

    report = kept.qc_report
    if args.report:
        report.to_csv(args.report, index=False)
        print(f"Wrote the quality control report to {args.report}")

    failed = report[~report["passed"]]
    print(f"{len(kept)} of {len(dataset)} series passed quality control.")
    if len(failed):
        print("\nExcluded:")
        with _wide_output():
            print(failed[["series_id", "reason"]].to_string(index=False))

    if args.out:
        kept.save(args.out)
        print(f"\nCopied the series that passed to {args.out}")
    return 0


def _run_radiomics(args) -> int:
    from .dataset import Dataset

    dataset = Dataset.from_directory(args.input)
    frame = dataset.radiomics(labels=args.labels, params=args.params, out_csv=args.out)
    feature_columns = [
        column for column in frame.columns if not column.startswith("diagnostics_")
    ]
    print(
        f"Extracted {len(feature_columns) - 1} features for {len(frame)} series "
        f"into {args.out}"
    )
    return 0


def _run_info(args) -> int:
    from .dataset import Dataset
    from .image import RadiologyImage

    if os.path.isdir(args.input) and not _looks_like_dicom_dir(args.input):
        dataset = Dataset.from_directory(args.input)
        frame = dataset.statistics()
        if args.json:
            print(frame.to_json(orient="records", indent=2))
        else:
            with _wide_output():
                print(frame.to_string(index=False))
        return 0

    image = RadiologyImage(args.input, mask=args.mask)
    stats = image.statistics()
    result = image.check()
    stats["quality_control"] = "pass" if result.passed else f"fail: {result.reason}"

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        width = max(len(key) for key in stats)
        for key, value in stats.items():
            print(f"{key:<{width}}  {value}")
    return 0


def _run_config(args) -> int:
    from .config import ProcessingConfig

    if args.preset == "radiomics":
        config = ProcessingConfig.radiomics(args.dataset)
    elif args.preset == "minimal":
        config = ProcessingConfig.minimal()
    elif args.dataset:
        config = ProcessingConfig.for_dataset(args.dataset)
    else:
        config = ProcessingConfig()

    if args.out:
        config.to_yaml(args.out)
        print(f"Wrote {args.out}")
    else:
        print(f"# {config.describe()}".replace("\n", "\n# "))
        print(config.to_yaml())
    return 0


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _config_from_args(args):
    from .config import ProcessingConfig

    if args.config:
        config = ProcessingConfig.load(args.config)
    elif args.dataset:
        config = ProcessingConfig.for_dataset(args.dataset)
    else:
        config = ProcessingConfig()

    overrides = {}
    for name in ("orient", "segment", "clip", "resample", "mask", "normalize"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    if getattr(args, "standardize_size", None) is not None:
        overrides["standardize_size"] = args.standardize_size
    if args.dimensionality:
        overrides["dimensionality"] = args.dimensionality
    if args.clip_min is not None:
        overrides["clip_min"] = args.clip_min
    if args.clip_max is not None:
        overrides["clip_max"] = args.clip_max
    if args.spacing:
        overrides["target_spacing"] = tuple(args.spacing)
    if args.shape:
        overrides["target_shape"] = tuple(args.shape)
    if args.organs:
        overrides["organs"] = list(args.organs)
        overrides.setdefault("segment", True)
    if args.output_format:
        overrides["output_format"] = args.output_format

    return config.replace(**overrides) if overrides else config


def _looks_like_dicom_dir(path: str) -> bool:
    for _, _, names in os.walk(path):
        if any(name.lower().endswith((".dcm", ".ima")) for name in names):
            return True
    return False


class _wide_output:
    """Let pandas use the full terminal width for a moment."""

    def __enter__(self):
        import pandas as pd

        self._context = pd.option_context(
            "display.max_rows", 200,
            "display.max_colwidth", 60,
            "display.width", 200,
        )
        self._context.__enter__()
        return self

    def __exit__(self, *exc_info):
        self._context.__exit__(*exc_info)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
