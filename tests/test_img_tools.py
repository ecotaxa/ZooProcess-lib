import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ZooProcess_lib.img_tools import crop, image_info, save_jpg_or_png_image


def test_crop_basic():
    """Test basic cropping functionality with a simple array."""
    # Create a test image (6x8 array with increasing values)
    test_image = np.arange(48).reshape(6, 8)

    # Test cropping a 3x4 section from the middle
    cropped = crop(test_image, top=1, left=2, bottom=4, right=6)

    # Expected result should be a 3x4 array
    expected = np.array([[10, 11, 12, 13], [18, 19, 20, 21], [26, 27, 28, 29]])

    np.testing.assert_array_equal(cropped, expected)
    assert cropped.shape == (3, 4)


def test_crop_edges():
    """Test cropping at the edges of the array."""
    test_image = np.ones((5, 5))

    # Test cropping from top-left corner
    top_left = crop(test_image, top=0, left=0, bottom=2, right=2)
    assert top_left.shape == (2, 2)

    # Test cropping to the edges
    full_height = crop(test_image, top=0, left=1, bottom=5, right=3)
    assert full_height.shape == (5, 2)


def test_crop_float_coords():
    """Test that float coordinates are properly converted to int."""
    test_image = np.ones((10, 10))

    # Float coordinates are floored:
    # top=1.8 -> 1, bottom=4.9 -> 4: difference = 3 rows
    # left=1.2 -> 1, right=6.1 -> 6: difference = 5 columns
    cropped = crop(test_image, top=1.8, left=1.2, bottom=4.9, right=6.1)
    assert cropped.shape == (3, 5)  # Shape is (bottom-top, right-left) after floor


def test_crop_input_validation():
    """Test that invalid inputs raise appropriate errors."""
    test_image = np.ones((5, 5))

    # Test with coordinates outside image bounds
    with pytest.raises(IndexError):
        crop(test_image, top=0, left=0, bottom=10, right=10)

    # Test with negative coordinates
    with pytest.raises(IndexError):
        crop(test_image, top=-1, left=0, bottom=3, right=3)


def test_save_jpg_or_png_image_roundtrip():
    """Test roundtrip for save_jpg_or_png_image, checking description and resolution."""
    # Create a simple test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[25:75, 25:75] = [255, 0, 0]  # Red square in the middle

    # Test parameters
    test_resolution = 300
    test_description = "Test image description"

    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with JPG format
        jpg_path = Path(temp_dir) / "test_image.jpg"
        save_jpg_or_png_image(test_image, test_resolution, jpg_path, test_description)

        # Load the JPG image and check metadata
        jpg_pil_image = Image.open(jpg_path)
        jpg_info = image_info(jpg_pil_image)

        # Verify resolution (allow for small floating-point differences)
        jpg_dpi_x, jpg_dpi_y = jpg_info["dpi"]
        assert abs(jpg_dpi_x - test_resolution) < 0.01
        assert abs(jpg_dpi_y - test_resolution) < 0.01

        # Verify description (in EXIF data for JPG)
        assert jpg_info.get("ImageDescription") == test_description

        # Test with PNG format
        png_path = Path(temp_dir) / "test_image.png"
        save_jpg_or_png_image(test_image, test_resolution, png_path, test_description)

        # Load the PNG image and check metadata
        png_pil_image = Image.open(png_path)
        png_info = image_info(png_pil_image)

        # Verify resolution (allow for small floating-point differences)
        png_dpi_x, png_dpi_y = png_info["dpi"]
        assert abs(png_dpi_x - test_resolution) < 0.01
        assert abs(png_dpi_y - test_resolution) < 0.01

        # Verify description (in text metadata for PNG)
        assert png_info.get("Description") == test_description
