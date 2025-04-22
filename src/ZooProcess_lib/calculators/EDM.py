import numpy as np

# Constants from the Java code
# These determine the scaling of the distance map
ONE = 41
SQRT2 = 58  # approx 41 * sqrt(2)
SQRT5 = 92  # approx 41 * sqrt(5)
MAX_VAL_16BIT = 32767  # Suitable large number for initialization


def make_16bit_edm(binary_image: np.ndarray) -> np.ndarray:
    """
    Calculates a 16-bit grayscale Euclidean Distance Map (EDM) for a binary
    NumPy array. Foreground pixels (non-zero) are initialized to a high value,
    background (zero) to 0. The output value is approximately ONE * distance.

    Uses the two-pass algorithm similar to the Java EDM plugin.

    Args:
        binary_image: 2D NumPy array (uint8 or bool) where 0 is background
                      and non-zero is foreground.

    Returns:
        2D NumPy array (int32) containing the 16-bit EDM.
    """
    if binary_image.ndim != 2:
        raise ValueError("Input image must be 2D")
    if binary_image.dtype != np.bool_ and not np.issubdtype(
        binary_image.dtype, np.integer
    ):
        print(
            f"Warning: Input image dtype is {binary_image.dtype}. Converting to bool."
        )
        binary_image = binary_image != 0  # Ensure boolean interpretation

    height, width = binary_image.shape
    # Use np.int32 for intermediate calculations to avoid potential overflow
    # Initialize foreground to MAX_VAL_16BIT, background to 0
    edm16 = np.where(binary_image, MAX_VAL_16BIT, 0).astype(np.int32)

    # --- Pass 1: Top-left to bottom-right ---
    # Process pixels based on already processed neighbors (top, left, top-left, etc.)
    # This requires careful handling of neighbors based on the specific kernel used.
    # The Java code uses a specific 5x5 neighborhood check in each pass.

    # Neighbors offsets relative to current pixel (y, x) for Pass 1 checks:
    # These correspond to the indices checked in setValue/setEdgeValue in the Java code
    # for the *first* pass (reading from already processed neighbors)
    neighbors_pass1 = [
        # Direct neighbors (distance ONE)
        (-1, 0),
        (0, -1),
        # Diagonal neighbors (distance SQRT2)
        (-1, -1),
        (-1, 1),
        # Knight's move neighbors (distance SQRT5)
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
    ]
    distances_pass1 = [ONE, ONE, SQRT2, SQRT2, SQRT5, SQRT5, SQRT5, SQRT5]

    for y in range(height):
        for x in range(width):
            if edm16[y, x] > 0:  # Only process foreground pixels
                min_dist = edm16[y, x]
                for i, (dy, dx) in enumerate(neighbors_pass1):
                    ny, nx = y + dy, x + dx
                    # Check boundary conditions
                    if 0 <= ny < height and 0 <= nx < width:
                        dist = edm16[ny, nx] + distances_pass1[i]
                        if dist < min_dist:
                            min_dist = dist
                edm16[y, x] = min_dist

    # --- Pass 2: Bottom-right to top-left ---
    # Process pixels based on already processed neighbors (bottom, right, etc.)
    # Neighbors offsets relative to current pixel (y, x) for Pass 2 checks:
    neighbors_pass2 = [
        # Direct neighbors (distance ONE)
        (1, 0),
        (0, 1),
        # Diagonal neighbors (distance SQRT2)
        (1, 1),
        (1, -1),
        # Knight's move neighbors (distance SQRT5)
        (2, -1),
        (2, 1),
        (1, -2),
        (1, 2),
    ]
    distances_pass2 = [ONE, ONE, SQRT2, SQRT2, SQRT5, SQRT5, SQRT5, SQRT5]

    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if edm16[y, x] > 0:  # Only process foreground pixels
                min_dist = edm16[y, x]
                for i, (dy, dx) in enumerate(neighbors_pass2):
                    ny, nx = y + dy, x + dx
                    # Check boundary conditions
                    if 0 <= ny < height and 0 <= nx < width:
                        dist = edm16[ny, nx] + distances_pass2[i]
                        if dist < min_dist:
                            min_dist = dist
                edm16[y, x] = min_dist

    return edm16


def convert_edm_to_bytes(edm16: np.ndarray) -> np.ndarray:
    """
    Converts the 16-bit scaled EDM to an 8-bit image by dividing by ONE.
    Args:
        edm16: 2D NumPy array (int32 or int16) from make_16bit_edm.
    Returns:
        2D NumPy array (uint8) representing the distance map (0-255).
    """
    if ONE <= 0:
        raise ValueError("ONE must be positive for scaling.")
    # Add half for rounding before integer division
    round_val = ONE // 2
    edm8 = (edm16 + round_val) // ONE
    edm8 = np.clip(edm8, 0, 255)  # Ensure values are within 8-bit range
    return edm8.astype(np.uint8)


def euclidean_distance_map(
    binary_image: np.ndarray, foreground_is_zero: bool = False
) -> np.ndarray:
    """
    Calculates the 8-bit Euclidean Distance Map.

    Args:
        binary_image: Input 2D NumPy array (binary).
        foreground_is_zero: If True, assumes 0 is foreground and inverts.
                            Otherwise, assumes non-zero is foreground.

    Returns:
        8-bit EDM NumPy array (uint8).
    """
    img = binary_image.copy()
    if foreground_is_zero:
        # Find max value if not boolean, assumes integer type
        max_val = 1 if img.dtype == np.bool_ else np.max(img)
        img = max_val - img  # Invert

    # Ensure internal representation is 0=background, non-zero=foreground
    img_proc = img != 0

    edm16 = make_16bit_edm(img_proc)
    edm8 = convert_edm_to_bytes(edm16)

    # Invert back if original had black background (non-zero foreground)
    # and the output convention should match input (e.g., distance FROM foreground)
    # However, EDM standard interpretation is distance TO nearest background.
    # The Java code inverts at the end based on Prefs.blackBackground.
    # Here, we return the standard EDM (high values away from background).
    # If inverted output is needed, invert edm8 here: edm8 = 255 - edm8

    return edm8
