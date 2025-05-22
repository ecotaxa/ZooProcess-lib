import tempfile
from pathlib import Path

from ZooProcess_lib.LegacyConfig import ProjectMeta
from .data_dir import PROJECT_DIR
from ZooProcess_lib.ZooscanFolder import ZooscanMetaFolder


def test_project_meta_write():
    """Test that ProjectMeta.write correctly writes a metadata.txt file in the same format as it is read."""
    # Path to the sample metadata.txt file
    metadata_path = (
        PROJECT_DIR / ZooscanMetaFolder.SUDIR_PATH / ZooscanMetaFolder.PROJECT_META
    )

    # Read the metadata file
    original_meta = ProjectMeta.read(metadata_path)

    # Create a temporary file to write the metadata to
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        # Write the metadata to the temporary file
        original_meta.write(temp_path)

        # Read the metadata back from the temporary file
        written_meta = ProjectMeta.read(temp_path)

        # Verify that all attributes in the original metadata are present in the written metadata
        for attr_name in dir(original_meta):
            # Skip special attributes, methods, and class variables
            if attr_name.startswith("__") or callable(getattr(original_meta, attr_name)) or attr_name == "__annotations__":
                continue

            # Get the attribute values
            original_value = getattr(original_meta, attr_name)
            written_value = getattr(written_meta, attr_name)

            # Skip default values for primitive types
            if original_value == "" or original_value == 0 or original_value == 0.0:
                continue

            # Verify that the attribute values are the same
            assert original_value == written_value, f"Attribute {attr_name} has different values: {original_value} != {written_value}"

        print("All tests passed!")
    finally:
        # Clean up the temporary file
        temp_path.unlink()
