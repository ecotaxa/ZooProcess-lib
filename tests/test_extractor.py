import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from ZooProcess_lib.Extractor import Extractor
from ZooProcess_lib.ROI import ROI
from ZooProcess_lib.img_tools import load_image, remove_footer_and_white_borders
from ZooProcess_lib.ImageJLike import images_difference


def test_extractor_border_and_inverse_transformation():
    """
    Test that adding a border to an image using Extractor and then removing it
    with remove_footer_and_white_borders results in the original image.
    """
    # Load the test image
    test_image_path = Path("data/images/measures/crop_13892_13563.png")
    original_image = load_image(test_image_path, cv2.IMREAD_GRAYSCALE)

    # Create a simple ROI that covers the entire image
    height, width = original_image.shape
    mask = np.ones((height, width), dtype=np.uint8)
    roi = ROI(x=0, y=0, mask=mask)

    # Create an Extractor instance with standard parameters
    extractor = Extractor(longline_mm=1.0, threshold=200)

    # Extract the image with the ROI (this should just return a copy of the original image)
    extracted_image = extractor.extract_image_at_ROI(original_image, roi, erasing_background=False)

    # Add border and legend to the image
    # Calculate longline value (1mm at 2400 dpi)
    longline = 1.0 * 2400 / 25.4  # Same calculation as in extract_all_with_border_to_dir
    image_with_border = extractor._add_border_and_legend(extracted_image, longline)

    # Apply the inverse transformation
    restored_image = remove_footer_and_white_borders(image_with_border)

    # Compare the original and restored images
    # After fixing the bug in remove_footer_and_white_borders, the shapes should be identical
    assert original_image.shape == restored_image.shape, f"Shapes differ: original {original_image.shape}, restored {restored_image.shape}"

    # Check if all pixels are identical
    diff = images_difference(original_image, restored_image)
    assert np.all(diff == 0), f"Images are not identical. Max difference: {np.max(diff)}"
