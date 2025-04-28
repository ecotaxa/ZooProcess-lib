from pathlib import Path

from ZooProcess_lib.ReadLog import LogReader
from ZooProcess_lib.ZooscanProject import ArchivedZooscanProject
from .env_fixture import projects
from .projects_for_test import ROND_CARRE


def test_readfile(projects):
    TP = ArchivedZooscanProject(projects, ROND_CARRE)

    scan_name = "test_01_tot"
    index = 1
    logfile = TP.getLogFile(scan_name, index)

    logfile_path = Path(logfile)
    assert logfile_path.is_file()
    assert logfile_path.stat().st_size > 0
    assert logfile_path.suffix == ".txt"
    assert logfile_path.name.startswith(scan_name)


def test_findBackgroundUsed(projects):
    TP = ArchivedZooscanProject(projects, ROND_CARRE)
    scan_name = "test_01_tot"
    logfile = TP.getLogFile(scan_name, 1)

    log = LogReader(logfile)
    background = log.getBackgroundPattern()

    assert background == "20141003_1144"
