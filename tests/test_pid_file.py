import tempfile
from pathlib import Path
from ZooProcess_lib.LegacyMeta import PidFile
from data_dir import WORK_DIR


def test_read_pid_file():
    # Path to the test .pid file
    pid_file_path = WORK_DIR / "apero2023_tha_bioness_006_st20_n_n7_d2_2_sur_4_1_dat1.pid"

    # Read the .pid file
    pid_file = PidFile.read(pid_file_path)

    # Test that sections were parsed correctly
    assert "Image" in pid_file.sections
    assert "Sample" in pid_file.sections
    assert "VueScan" in pid_file.sections

    # Test that values can be retrieved correctly
    assert pid_file.get_value("Image", "Scanning_date") == "20240321_1049"
    assert pid_file.get_value("Sample", "SampleId") == "apero2023_tha_bioness_006_st20_n_n7"

    # Test that the data section was parsed correctly
    assert len(pid_file.header_row) > 0
    assert len(pid_file.data_rows) > 0

    # Check a specific data row
    if pid_file.data_rows:
        first_row = pid_file.data_rows[0]
        assert "Item" in first_row
        assert "Label" in first_row
        assert "Area" in first_row


def test_pid_file_roundtrip():
    """
    Test that a PID file can be read, written, and read again without losing any data.
    This ensures the write method correctly preserves all information from the original file.
    """
    # Path to the test .pid file
    original_pid_file_path = WORK_DIR / "apero2023_tha_bioness_006_st20_n_n7_d2_2_sur_4_1_dat1.pid"

    # Read the original .pid file
    original_pid_file = PidFile.read(original_pid_file_path)

    # Create a temporary file for the roundtrip test
    with tempfile.NamedTemporaryFile(suffix='.pid', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        # Write the PID file to the temporary path
        original_pid_file.write(tmp_path)

        # Read the written file back
        reread_pid_file = PidFile.read(tmp_path)

        # Compare sections
        assert original_pid_file.sections.keys() == reread_pid_file.sections.keys()
        for section_name in original_pid_file.sections:
            original_section = original_pid_file.sections[section_name]
            reread_section = reread_pid_file.sections[section_name]
            assert original_section.keys() == reread_section.keys()
            for key in original_section:
                assert original_section[key] == reread_section[key]

        # Compare header row
        assert original_pid_file.header_row == reread_pid_file.header_row

        # Compare data rows
        assert len(original_pid_file.data_rows) == len(reread_pid_file.data_rows)
        for i, original_row in enumerate(original_pid_file.data_rows):
            reread_row = reread_pid_file.data_rows[i]
            assert original_row.keys() == reread_row.keys()
            for key in original_row:
                assert original_row[key] == reread_row[key]

    finally:
        # Clean up the temporary file
        tmp_path.unlink(missing_ok=True)
