import dataclasses

from ZooProcess_lib.LegacyConfig import ProjectMeta
from ZooProcess_lib.ZooscanFolder import ZooscanMetaFolder
from data_dir import PROJECT_DIR


def test_project_meta_read():
    """Test that ProjectMeta.read correctly reads a metadata.txt file and uses predefined types."""
    # Path to the sample metadata.txt file
    metadata_path = (
        PROJECT_DIR / ZooscanMetaFolder.SUDIR_PATH / ZooscanMetaFolder.PROJECT_META
    )

    # Read the metadata file
    meta = ProjectMeta.read(metadata_path)

    # Print out all attributes and their types
    print("ProjectMeta attributes:")
    for attr_name in dir(meta):
        # Skip special attributes and methods
        if attr_name.startswith("__") or callable(getattr(meta, attr_name)):
            continue

        attr_value = getattr(meta, attr_name)
        print(f"{attr_name}: {attr_value} ({type(attr_value).__name__})")

    # Verify that all explicitly defined fields are present and have the correct types
    for field in dataclasses.fields(ProjectMeta):
        assert hasattr(meta, field.name), f"Field {field.name} is missing"

        # Skip fields that aren't in the metadata file
        if (
            getattr(meta, field.name) == ""
            or getattr(meta, field.name) == -1
            or getattr(meta, field.name) == -1.0
        ):
            continue

        assert isinstance(
            getattr(meta, field.name), field.type
        ), f"Field {field.name} has wrong type"

    # Verify some specific attributes
    assert hasattr(meta, "SampleId")
    assert isinstance(meta.SampleId, str)
    assert meta.SampleId == "apero2023_tha_bioness_sup2000_017_st66_d_n1"

    assert hasattr(meta, "Latitude")
    assert isinstance(meta.Latitude, float)
    assert meta.Latitude == 51.4322000

    assert hasattr(meta, "Depth")
    assert isinstance(meta.Depth, float)
    assert meta.Depth == 99999.0

    # Verify that fields not explicitly defined are still added as strings
    # (This is for backward compatibility)
    if hasattr(meta, "some_undefined_field"):
        assert isinstance(meta.some_undefined_field, str)

    print("All tests passed!")


def test_zooscan_meta_folder():
    """Test that ZooscanMetaFolder correctly reads all metadata files."""
    # Create a ZooscanMetaFolder instance
    meta_folder = ZooscanMetaFolder(PROJECT_DIR)

    # Test read_project_meta
    project_meta = meta_folder.read_project_meta()
    assert isinstance(project_meta, ProjectMeta)
    assert project_meta.SampleId == "apero2023_tha_bioness_sup2000_017_st66_d_n1"
    assert project_meta.Ship == "thalassa"
    assert project_meta.Latitude == 51.4322000
    assert project_meta.Longitude == 18.3108000

    # Test read_samples_table
    samples_table = meta_folder.read_samples_table()
    assert isinstance(samples_table, list)
    assert len(samples_table) > 0
    assert isinstance(samples_table[0], dict)

    # Check CSV header structure (all expected keys are present)
    expected_sample_headers = [
        "sampleid", "ship", "scientificprog", "stationid", "date", "latitude", "longitude",
        "depth", "ctdref", "otherref", "townb", "towtype", "nettype", "netmesh", "netsurf",
        "zmax", "zmin", "vol", "sample_comment", "vol_qc", "depth_qc", "sample_qc", "barcode",
        "latitude_end", "longitude_end", "net_duration", "ship_speed_knots", "cable_length",
        "cable_angle", "cable_speed", "nb_jar"
    ]
    for header in expected_sample_headers:
        assert header in samples_table[0], f"Header '{header}' missing from samples table"

    # Check a specific sample
    sample_017_st66 = next(
        (
            s
            for s in samples_table
            if s["sampleid"] == "apero2023_tha_bioness_sup2000_017_st66_d_n1"
        ),
        None,
    )
    assert sample_017_st66 is not None
    assert sample_017_st66["ship"] == "thalassa"
    assert sample_017_st66["scientificprog"] == "apero"
    assert sample_017_st66["stationid"] == "66"
    assert sample_017_st66["latitude"] == "51.5293"
    assert sample_017_st66["longitude"] == "19.2159"

    # Test read_scans_table
    scans_table = meta_folder.read_scans_table()
    assert isinstance(scans_table, list)
    assert len(scans_table) > 0
    assert isinstance(scans_table[0], dict)

    # Check CSV header structure (all expected keys are present)
    expected_scan_headers = [
        "scanid", "sampleid", "scanop", "fracid", "fracmin", "fracsup", "fracnb",
        "observation", "code", "submethod", "cellpart", "replicates", "volini", "volprec"
    ]
    for header in expected_scan_headers:
        assert header in scans_table[0], f"Header '{header}' missing from scans table"

    # Check a specific scan
    scan_017_st66 = next(
        (
            s
            for s in scans_table
            if s["sampleid"] == "apero2023_tha_bioness_sup2000_017_st66_d_n1"
            and s["fracid"] == "d2_4_sur_4"
        ),
        None,
    )
    assert scan_017_st66 is not None
    assert (
        scan_017_st66["scanid"]
        == "apero2023_tha_bioness_sup2000_017_st66_d_n1_d2_4_sur_4_1"
    )
    assert scan_017_st66["scanop"] == "adelaide_perruchon"
    assert scan_017_st66["fracmin"] == "2000"
    assert scan_017_st66["fracsup"] == "999999"
    assert scan_017_st66["fracnb"] == "4"

    print("All ZooscanMetaFolder tests passed!")
