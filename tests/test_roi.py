import numpy as np
import pytest

from ZooProcess_lib.ROI import ROI, unique_visible_key, box_from_key


def test_unique_visible_key():
    """Test that unique_visible_key generates the expected string format."""
    # Create a simple ROI
    mask = np.ones((40, 30), dtype=np.uint8)  # height=40, width=30
    roi = ROI(x=10, y=20, mask=mask)
    
    # Generate the key
    key = unique_visible_key(roi)
    
    # Check the format and values
    assert key == "xAy14w1Eh28"  # 10 in hex is A, 20 is 14, 30 is 1E, 40 is 28


def test_box_from_key():
    """Test that box_from_key correctly parses the key string."""
    # Test with a simple key
    key = "xAy14w1Eh28"
    x, y, width, height = box_from_key(key)
    
    assert x == 10
    assert y == 20
    assert width == 30
    assert height == 40


def test_roundtrip_conversion():
    """Test that converting from ROI to key and back gives the original values."""
    # Create a simple ROI
    mask = np.ones((40, 30), dtype=np.uint8)  # height=40, width=30
    roi = ROI(x=10, y=20, mask=mask)
    
    # Convert to key and back
    key = unique_visible_key(roi)
    x, y, width, height = box_from_key(key)
    
    # Check that we got the original values back
    assert x == roi.x
    assert y == roi.y
    assert width == roi.mask.shape[1]
    assert height == roi.mask.shape[0]


def test_box_from_key_with_large_values():
    """Test that box_from_key works with large hexadecimal values."""
    # Test with large values
    key = "x1ABCy2DEFw3456h789A"
    x, y, width, height = box_from_key(key)
    
    assert x == 0x1ABC
    assert y == 0x2DEF
    assert width == 0x3456
    assert height == 0x789A


def test_box_from_key_with_zero_values():
    """Test that box_from_key works with zero values."""
    # Test with zero values
    key = "x0y0w0h0"
    x, y, width, height = box_from_key(key)
    
    assert x == 0
    assert y == 0
    assert width == 0
    assert height == 0