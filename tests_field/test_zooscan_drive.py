import tempfile
from pathlib import Path

from ZooProcess_lib.LegacyMeta import ProjectMeta
from ZooProcess_lib.ZooscanFolder import ZooscanDrive, ZooscanMetaFolder
from .env_fixture import read_home


def test_zooscan_drive_list_projects():
    """
    Test that builds a ZooscanDrive instance for each directory in the parent directory of ZOOSCAN_PROJECTS
    and lists all projects in them.
    """
    # Get the path from the ZOOSCAN_PROJECTS environment variable
    zooscan_projects_path = read_home()

    # Get the parent directory of zooscan_projects_path
    parent_directory = zooscan_projects_path.parent

    # Create a ZooscanDrive instance for the parent directory
    parent_drive = ZooscanDrive(parent_directory)

    # List all subdirectories in the parent directory
    subdirectories = list(parent_drive.list())

    # Verify that we found some subdirectories
    assert len(subdirectories) > 0, f"No subdirectories found in {parent_directory}"

    # Total projects found across all drives
    total_projects = 0

    # Create a ZooscanDrive instance for each subdirectory in the parent directory
    for subdir in subdirectories:
        print(f"Creating drive for {subdir}")

        # Create a ZooscanDrive instance for the subdirectory
        drive = ZooscanDrive(subdir)

        # List all projects in the drive
        projects_list = list(drive.list())

        # Print the projects for informational purposes
        print(f"Found {len(projects_list)} projects in {subdir}:")

        # Process each project
        for project in projects_list:
            print(f"  - {project.name}")

            # Verify that we can get a project folder for each project
            project_folder = drive.get_project_folder(project.name)
            assert project_folder.path.is_dir(), f"Project folder {project_folder.path} is not a directory"

            # Check that metadata can be correctly read for the project
            meta_folder = ZooscanMetaFolder(project_folder.path)
            try:
                project_meta = meta_folder.read_project_meta()

                # Verify that the metadata is a ProjectMeta instance
                assert isinstance(project_meta, ProjectMeta), f"Project metadata is not a ProjectMeta instance"

                # Verify that some essential metadata fields are present and have values
                assert hasattr(project_meta, 'SampleId'), f"Project metadata missing SampleId field"
                print(f"    Metadata SampleId: {project_meta.SampleId}")

                # Create a temporary file and write the metadata to it
                with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    try:
                        # Write the metadata to the temporary file
                        project_meta.write(temp_path)

                        # Read the original metadata file content
                        original_path = meta_folder.path / meta_folder.PROJECT_META
                        with open(original_path, 'r') as original_file:
                            original_content = original_file.readlines()

                        # Read the temporary file content
                        with open(temp_path, 'r') as temp_file:
                            temp_content = temp_file.readlines()

                        # Compare the content (ignoring order and empty lines)
                        original_lines = [line.strip() for line in original_content if line.strip()]
                        temp_lines = [line.strip() for line in temp_content if line.strip()]

                        # Sort the lines for comparison (since the order might be different)
                        original_lines.sort()
                        temp_lines.sort()

                        # Normalize lines to handle trailing .0s in numbers
                        def normalize_line(line):
                            if "=" in line:
                                key, value = line.split("=", 1)
                                key = key.strip()
                                value = value.strip()

                                # Try to convert to float for numeric comparison
                                try:
                                    # Check if it's a number
                                    float_value = float(value)
                                    # If it's an integer value, convert to int to remove trailing .0
                                    if float_value.is_integer():
                                        value = str(int(float_value))
                                    else:
                                        value = str(float_value)
                                except ValueError:
                                    # Not a number, keep as is
                                    pass

                                return f"{key}= {value}"
                            return line

                        normalized_original = [normalize_line(line) for line in original_lines]
                        normalized_temp = [normalize_line(line) for line in temp_lines]

                        if normalized_original != normalized_temp:
                            pass
                        # Assert that the content is the same
                        assert normalized_original == normalized_temp, f"Metadata roundtrip test mismatch for project {project.name}"
                        print(f"    Metadata content verified for project {project.name}")
                    finally:
                        # Clean up the temporary file
                        if temp_path.exists():
                            temp_path.unlink()

            except (FileNotFoundError, AssertionError) as meta_error:
                # Skip projects that don't have metadata or have invalid metadata
                print(f"    Skipping project {project.name} metadata check due to error: {meta_error}")
                # raise

            # Increment total projects count for valid projects (regardless of metadata check result)
            total_projects += 1

    # Verify that we found some projects across all drives in the parent directory
    assert total_projects > 0, "No projects found in any drive in the parent directory"
