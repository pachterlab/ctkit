"""Tests for argument validation on the public methods."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import ctkit
from ctkit import Dataset, RadiologyImage


class TestRejects:
    """A wrong argument fails at the call, not several steps later."""

    def test_a_string_where_a_number_belongs(self, image):
        with pytest.raises(ValidationError):
            image.clip("cold", 300)

    def test_a_number_where_a_sequence_belongs(self, image):
        with pytest.raises(ValidationError):
            image.resample(0.8)

    def test_a_string_where_an_index_belongs(self, image):
        with pytest.raises(ValidationError):
            image.select_slice(mode="index", index="middle")

    def test_the_error_names_the_method_and_the_argument(self, image):
        with pytest.raises(ValidationError) as raised:
            image.standardize_size(x="wide")
        message = str(raised.value)
        assert "standardize_size" in message
        assert "x" in message

    def test_a_cohort_fails_at_the_call_not_at_the_read(self, cohort_dir):
        """The whole point of validating the Dataset method too: steps are deferred."""
        with pytest.raises(ValidationError):
            Dataset(str(cohort_dir)).clip("cold", 300)

    def test_a_function_fails_before_the_cohort_is_discovered(self, tmp_path):
        with pytest.raises(ValidationError):
            ctkit.clip(str(tmp_path / "does-not-exist"), "cold")


class TestAccepts:
    """Validation must not reject what a scientific caller actually has."""

    def test_numpy_scalars(self, image):
        image.clip(np.float32(-200), np.int64(300))
        assert image.array.min() >= -200

        image.standardize_size(np.int64(24), np.int64(24), np.int64(8))
        assert image.shape == (24, 24, 8)

    def test_a_numpy_array_as_a_spacing(self, image):
        image.resample(np.array([1.0, 1.0, 3.0]))
        assert image.spacing == pytest.approx((1.0, 1.0, 3.0))

    def test_a_numpy_array_of_labels(self, image):
        image.apply_mask(labels=np.array([1, 2]))
        assert image.shape != (0, 0, 0)

    def test_a_path_object(self, image, tmp_path):
        image.save(tmp_path / "case.nii.gz")
        assert (tmp_path / "case.nii.gz").exists()

    def test_a_path_object_for_a_cohort(self, cohort_dir, tmp_path):
        written = Dataset(str(cohort_dir)).save(tmp_path / "out", progress=False)
        assert len(written) == 6

    def test_lists_and_tuples_alike(self, image):
        image.resample([1.0, 1.0, 3.0])
        assert image.spacing == pytest.approx((1.0, 1.0, 3.0))


class TestOptions:
    """Arguments with a fixed set of options carry that set in their type."""

    @pytest.mark.parametrize(
        "call, allowed",
        [
            (lambda image: image.select_slice(mode="brightest"), "'mask' or 'index'"),
            (lambda image: image.resample((1.0, 1.0, 3.0), interpolator="cubic"),
             "'linear', 'nearest' or 'bspline'"),
            (lambda image: image.normalize(method="cohort"), "'volume' or 'dataset'"),
            (lambda image: image.check(level="headers"),
             "'metadata', 'volume' or 'all'"),
            (lambda image: image.save("out", output_format="dicom"),
             "'nifti', 'nii', 'numpy' or 'npy'"),
        ],
    )
    def test_the_message_lists_the_options(self, image, call, allowed):
        with pytest.raises(ValidationError, match=allowed):
            call(image)

    def test_every_listed_output_format_is_writable(self, image, tmp_path):
        from ctkit.validation import OUTPUT_FORMATS

        for output_format in OUTPUT_FORMATS:
            written = image.save(tmp_path / f"case_{output_format}", output_format=output_format)
            assert written.endswith((".nii", ".nii.gz", ".npy"))

    def test_dicom_is_read_only_and_says_so(self, image, tmp_path):
        from ctkit.io import save_image

        with pytest.raises(ValueError, match="DICOM output is not supported"):
            save_image(image.image, str(tmp_path / "case"), output_format="dicom")

    def test_a_dicom_extension_is_replaced_not_appended(self, image, tmp_path):
        written = image.save(tmp_path / "case.dcm")
        assert written.endswith("case.nii.gz")


class TestUnchanged:
    """Rules spanning several arguments stay in the methods, with their guidance."""

    def test_multi_argument_rules_still_explain_what_to_do(self, image):
        with pytest.raises(ValueError, match="needs index="):
            image.select_slice(mode="index")

        with pytest.raises(ValueError, match="pooled over the cohort"):
            image.normalize(method="dataset")

    def test_signatures_survive_for_help_and_editors(self):
        import inspect

        signature = inspect.signature(RadiologyImage.clip)
        assert list(signature.parameters) == ["self", "min_value", "max_value"]
        assert RadiologyImage.clip.__doc__.startswith("Clamp intensities")

    def test_only_the_public_methods_are_wrapped(self):
        assert hasattr(RadiologyImage.clip, "__wrapped__")
        assert not hasattr(RadiologyImage._resolve_mask_label, "__wrapped__")

    def test_no_getattr_hook_hiding_typos_from_type_checkers(self):
        """``filter`` is a real method, so mypy still catches every other typo."""
        assert "__getattr__" not in vars(RadiologyImage)
        assert "filter" in vars(RadiologyImage)

    def test_the_functional_form_is_validated_too(self, image):
        with pytest.raises(ValidationError, match="clip"):
            ctkit.clip(image, "cold")
