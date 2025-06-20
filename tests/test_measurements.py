import tempfile
from pathlib import Path

from ZooProcess_lib.Features import TYPE_BY_LEGACY
from ZooProcess_lib.LegacyMeta import Measurements
from .data_dir import WORK_DIR


def test_read_measurements():
    """Test reading a measurement file."""
    # Path to the test measurement file
    meas_file = WORK_DIR / "s_17_1_tot_1_meas.txt"

    # Read the measurements
    measurements = Measurements.read(meas_file, TYPE_BY_LEGACY)

    # Test that the measurements were read correctly
    assert len(measurements.get_data_rows()) > 0
    assert len(measurements.get_header_row()) > 0

    # Check the first measurement
    first_measurement = measurements.get_data_rows()[0]
    assert "Area" in first_measurement
    assert "Mean" in first_measurement
    assert "StdDev" in first_measurement

    # Check specific values from the first measurement
    assert first_measurement["Area"] == 803
    assert abs(first_measurement["Mean"] - 228.267) < 0.001
    assert abs(first_measurement["StdDev"] - 12.301) < 0.001


def test_measurements_roundtrip():
    """
    Test that a measurement file can be read, written, and read again without losing any data.
    This ensures the write method correctly preserves all information from the original file.
    """
    # Path to the test measurement file
    original_meas_file = WORK_DIR / "s_17_1_tot_1_meas.txt"

    # Read the original measurement file
    original_measurements = Measurements.read(original_meas_file, TYPE_BY_LEGACY)

    # Create a temporary file for the roundtrip test
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        # Write the measurements to the temporary path
        original_measurements.write(tmp_path)

        # Read the written file back
        reread_measurements = Measurements.read(tmp_path, TYPE_BY_LEGACY)

        # Compare header row
        assert (
            original_measurements.get_header_row()
            == reread_measurements.get_header_row()
        )

        # Compare data rows
        assert len(original_measurements.get_data_rows()) == len(
            reread_measurements.get_data_rows()
        )

        # Check a few specific measurements to ensure they match
        for i in range(min(5, len(original_measurements.get_data_rows()))):
            original_row = original_measurements.get_data_rows()[i]
            reread_row = reread_measurements.get_data_rows()[i]

            # Check that the keys match
            assert original_row.keys() == reread_row.keys()

            # Check that the values match for a few key fields
            for key in ["Area", "Mean", "StdDev", "BX", "BY", "Width", "Height"]:
                if isinstance(original_row[key], float):
                    assert abs(original_row[key] - reread_row[key]) < 0.001
                else:
                    assert original_row[key] == reread_row[key]

    finally:
        # Clean up the temporary file
        tmp_path.unlink(missing_ok=True)
