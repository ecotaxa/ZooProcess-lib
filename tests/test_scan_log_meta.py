from ZooProcess_lib.ZooscanFolder import ZooscanProjectFolder
from ZooProcess_lib.LegacyMeta import ScanLog
from data_dir import PROJECT_DIR, BACK_TIME


def test_get_log_meta_reads_expected_fields(monkeypatch):
    # Arrange: avoid reading missing config files by mocking the config reader
    from ZooProcess_lib.LegacyConfig import ZooscanConfig
    from ZooProcess_lib import LegacyMeta

    dummy_conf = ZooscanConfig(
        background_process="last",
        minsizeesd_mm=0.001,
        maxsizeesd_mm=0.001,
        upper=243,
        resolution=2400,
        longline_mm=0.001,
    )
    # Patch the classmethod to return our dummy configuration regardless of path
    monkeypatch.setattr(
        ZooscanConfig,
        "read",
        classmethod(lambda cls, path: dummy_conf),
        raising=True,
    )

    # Patch LegacyMeta.LogFile.read to return a crafted minimal object with expected values
    class _FakeLog(LegacyMeta.ScanLogFile):
        def __init__(self):
            super().__init__()
            self.sections = {
                "Image": {
                    "Scanning_date": "20240529_1541",
                    "Scanning_area": "large",
                    "Vuescan_version": "9.7.67",
                },
                "Input": {"Source": "PerfectionV700"},
                "Info": {
                    "Hardware": "Hydroptic_V3",
                    "Software": "9.7.67",
                    "Resolution": "2400",
                },
                "Image_Process": {
                    "Background_correct_using": str(BACK_TIME) + "_rest"
                },
            }

    monkeypatch.setattr(LegacyMeta.ScanLogFile, "read", classmethod(lambda cls, path: _FakeLog()))

    # Point to the sample project directory shipped with tests
    folder = ZooscanProjectFolder(PROJECT_DIR, PROJECT_DIR.name)

    # The tests' sample project uses 'raw' directory at project root
    # while the library expects 'Zooscan_scan/_raw'. Point raw.path to tests data.
    folder.zooscan_scan.raw.path = PROJECT_DIR / "raw"

    # Sample name and index derived from the provided fixture files
    sample = "apero2023_tha_bioness_017_st66_d_n1_d3"
    index = 1

    # Act
    # The log file is stored alongside raw images, not in the work directory
    log_meta = folder.zooscan_scan.raw.get_scan_log(sample, index)

    # Assert basic presence and type
    assert isinstance(log_meta, ScanLog)

    # Assert a few key fields parsed from the test log file
    # [Image]
    assert log_meta.scanning_date == "20240529_1541"
    assert log_meta.scanning_area == "large"
    assert log_meta.vuescan_version == "9.7.67"

    # [Input]
    assert log_meta.scanner_source == "PerfectionV700"

    # [Info]
    assert log_meta.info_hardware == "Hydroptic_V3"
    assert log_meta.info_software == "9.7.67"
    assert log_meta.info_resolution == 2400

    # Derived background timestamp from [Image_Process]/Background_correct_using
    assert log_meta.background_pattern == BACK_TIME

    # And we should have access to raw sections as well
    assert "Image_Process" in log_meta.sections
