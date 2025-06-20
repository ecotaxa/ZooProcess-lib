import cv2
import pytest

from ZooProcess_lib.Extractor import Extractor
from ZooProcess_lib.Features import BOX_MEASUREMENTS
from ZooProcess_lib.LegacyMeta import PidFile, Measurements
from ZooProcess_lib.ZooscanFolder import ZooscanProjectFolder, WRK_PID, WRK_JPGS, WRK_MEAS
from ZooProcess_lib.img_tools import load_image
from .env_fixture import projects
from .projects_repository import tested_samples


@pytest.mark.parametrize(
    "project, sample",
    tested_samples,
    ids=[sample for (_prj, sample) in tested_samples],
)
def test_can_read_pids(projects, project, sample):
    """
    Test that PID files can be read for all projects/samples.
    This test loops over all projects/samples and reads PidFile instances.
    """
    folder = ZooscanProjectFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    # Read PID file
    work_files = folder.zooscan_scan.work.get_files(sample, index)
    pid_file = work_files.get(WRK_PID)
    if pid_file is not None:
        # Read the PID file
        pid_file = PidFile.read(pid_file)

        # Test that sections were parsed correctly
        assert "Image" in pid_file.sections
        assert "Sample" in pid_file.sections

        # Test that values can be retrieved correctly
        assert pid_file.get_value("Sample", "SampleId") is not None

        # Test that the data section was parsed correctly
        assert len(pid_file.header_row) > 0
        assert len(pid_file.data_rows) > 0

        # Check a specific data row
        if pid_file.data_rows:
            for a_row in pid_file.data_rows:
                assert a_row["Tag"] == "1"
            first_row = pid_file.data_rows[0]
            assert "Item" in first_row
            assert "Label" in first_row
            assert "Area" in first_row


@pytest.mark.parametrize(
    "project, sample",
    tested_samples,
    ids=[sample for (_prj, sample) in tested_samples],
)
def test_jpegs_are_in_pids(projects, project, sample):
    """
    Consistency b/w jpeg vignettes and PID contents
    """
    folder = ZooscanProjectFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    work_files = folder.zooscan_scan.work.get_files(sample, index)
    # Read PID file path
    pid_file_path = work_files.get(WRK_PID)
    if pid_file_path is None:
        return
    # Read the PID file
    pid_file = PidFile.read(pid_file_path)

    longline_mm = 1.0
    longline = longline_mm * folder.zooscan_config.read().resolution / 25.4

    # Check that for each .jpg in work directory, a line exists in data_rows with its name
    assert WRK_JPGS in work_files

    jpg_files = work_files[WRK_JPGS]

    # Collect all jpg files that don't have a matching data row or have incorrect dimensions
    problems = []
    found = False
    for jpg_file in jpg_files:
        if "_color_" in jpg_file.name:
            continue
        # Check if the jpg file name contains any part of any Label value
        for row in pid_file.data_rows:
            num = row["Item"]
            label = row["Label"]
            if f"{label}_{num}.jpg" == jpg_file.name:
                found = True
                width, height = int(row["Width"]), int(row["Height"])
                height, width = Extractor.get_final_dimensions(
                    height, width, longline
                )
                # Also check if the jpg file dimensions match the BX and BY columns
                # Open the jpg file and get its dimensions
                img = load_image(jpg_file, cv2.IMREAD_GRAYSCALE)
                # img = remove_footer_and_white_borders(img) # No, it's JPEG so damaged
                img_height, img_width = img.shape
                if width != img_width or height != img_height:
                    delta_w, delta_h = img_width - width, img_height - height
                    problems.append(
                        f"{jpg_file.name} dimensions ({width}x{height}) don't match (from row) ({img_width}x{img_height}), delta={delta_w}x{delta_h}"
                    )

        if not found:
            problems.append(f"Not matched: {jpg_file.name}")

    # Assert only once at the end of the test
    assert not problems, ", ".join(problems)


@pytest.mark.parametrize(
    "project, sample",
    tested_samples,
    ids=[sample for (_prj, sample) in tested_samples],
)
def test_data_same_as_meas(projects, project, sample):
    """
    Compare data section of the PID with the WRK_MEAS read in same work folder
    """
    folder = ZooscanProjectFolder(projects, project)
    index = 1  # TODO: should come from get_names() below
    work_files = folder.zooscan_scan.work.get_files(sample, index)

    # Read PID file path
    pid_file_path = work_files.get(WRK_PID)
    if pid_file_path is None:
        return

    # Read the PID file
    pid_file = PidFile.read(pid_file_path)

    # Check if MEAS file exists
    meas_file_path = work_files.get(WRK_MEAS)
    if meas_file_path is None:
        return

    # Read the MEAS file
    meas_file = Measurements.read(meas_file_path)

    # Compare data rows from PID and MEAS files
    problems = []

    # Check if the number of rows is the same
    if len(pid_file.data_rows) != len(meas_file.data_rows):
        problems.append(f"Number of rows in PID ({len(pid_file.data_rows)}) doesn't match MEAS ({len(meas_file.data_rows)})")

    # Check if the data in each row matches
    for i, (pid_row, meas_row) in enumerate(zip(pid_file.data_rows, meas_file.data_rows)):
        # Compare only BOX_MEASUREMENTS fields, there is extreme rounding for all the rest in the PID file
        for field in BOX_MEASUREMENTS:
            if field in pid_row and field in meas_row:
                # Convert values to strings for comparison
                pid_value = str(pid_row[field])
                meas_value = str(meas_row[field])

                # Some values might be stored as floats in one file and integers in another
                # Try to normalize them for comparison
                try:
                    # Try to convert both values to floats for comparison
                    pid_float = float(pid_value)
                    meas_float = float(meas_value)

                    # Check if the values are integers stored as floats
                    pid_is_int = pid_float.is_integer()
                    meas_is_int = meas_float.is_integer()

                    # If both are integers, compare them as integers
                    if pid_is_int and meas_is_int:
                        if int(pid_float) != int(meas_float):
                            problems.append(f"Row {i+1}, field '{field}': PID value '{pid_value}' doesn't match MEAS value '{meas_value}'")
                    else:
                        # For floating point values, use a larger tolerance to account for rounding differences
                        # The tolerance is relative to the scale of the values
                        tolerance = max(1e-2, abs(pid_float) * 0.01)  # 1% relative tolerance or 0.01 absolute, whichever is larger

                        if abs(pid_float - meas_float) > tolerance:
                            # Check if the difference is due to rounding to 2 decimal places
                            pid_rounded = round(pid_float, 2)
                            meas_rounded = round(meas_float, 2)

                            if pid_rounded != meas_rounded:
                                problems.append(f"Row {i+1}, field '{field}': PID value '{pid_value}' doesn't match MEAS value '{meas_value}'")
                except (ValueError, TypeError):
                    # If conversion fails, compare as strings
                    if pid_value != meas_value:
                        problems.append(f"Row {i+1}, field '{field}': PID value '{pid_value}' doesn't match MEAS value '{meas_value}'")

    # Assert only once at the end of the test
    assert not problems, ", ".join(problems)
