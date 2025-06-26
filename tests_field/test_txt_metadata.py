import random

import pytest

from ZooProcess_lib.ZooscanFolder import ZooscanProjectFolder
from ZooProcess_lib.LegacyMeta import ProjectMeta, ScanMeta
from .env_fixture import projects
from .projects_repository import tested_samples

# The test takes ages, by randomizing the order there are better chances to see problems early
shuffled = tested_samples.copy()
random.shuffle(shuffled)

@pytest.mark.parametrize(
    "project, sample",
    shuffled,
    ids=[sample for (_prj, sample) in shuffled],
)
def test_read_project_and_scan_meta(projects, project, sample, tmp_path):
    """Test that we can read both ProjectMeta and ScanMeta for all samples in tested_samples"""
    folder = ZooscanProjectFolder(projects, project)
    index = 1  # Always a single scan

    # Read ProjectMeta
    project_meta = folder.zooscan_meta.read_project_meta()
    assert isinstance(project_meta, ProjectMeta)

    # Read ScanMeta
    scan_meta = folder.zooscan_scan.work.get_txt_meta(sample, index)

    # Verify that project metadata has expected fields
    assert hasattr(project_meta, 'SampleId')

    # Convert project metadata to dictionary
    _project_meta_dict = project_meta.to_dict()

    # If scan metadata exists, verify it
    if scan_meta is not None:
        assert isinstance(scan_meta, ScanMeta)
        assert hasattr(scan_meta, 'SampleId')

        # Convert scan metadata to dictionary
        _scan_meta_dict = scan_meta.to_dict()
