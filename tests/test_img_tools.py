import numpy as np
import pytest

from ZooProcess_lib.img_tools import crop


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
