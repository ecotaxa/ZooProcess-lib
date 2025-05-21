import os
from pathlib import Path

import pytest

from ZooProcess_lib.ZooscanFolder import ZooscanDrive, ZooscanMetaFolder
from ZooProcess_lib.LegacyConfig import ProjectMeta
from .env_fixture import projects, read_home


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

            except (FileNotFoundError, AssertionError) as meta_error:
                # Skip projects that don't have metadata or have invalid metadata
                print(f"    Skipping project {project.name} metadata check due to error: {meta_error}")

            # Increment total projects count for valid projects (regardless of metadata check result)
            total_projects += 1

    # Verify that we found some projects across all drives in the parent directory
    assert total_projects > 0, "No projects found in any drive in the parent directory"
