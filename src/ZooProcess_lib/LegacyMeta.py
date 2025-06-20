# Various file aside from graphical data
import csv
import dataclasses
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Type


class LutFile:
    """A .lut file contains processing directives for a whole project"""

    def __init__(self):
        self.min: int = 0
        self.max: int = 65536
        self.gamma: float = 1
        self.sens: str = "before"
        # Appeared in a later version
        self.adjust: str = "no"
        self.odrange: float = 1.8
        # Appeared in a later version
        self.ratio: float = 1.15
        # Appeared in a later version
        self.sizelimit: int = 800  # Unused?
        self.overlap: float = 0.07
        # Appeared in a later version
        self.medianchoice: str = "no"
        self.medianvalue: int = 1
        # Appeared in a later version
        self.resolutionreduct: int = 1200

    @staticmethod
    def read(path: Path) -> "LutFile":
        ret = LutFile()
        with open(path, "r") as f:
            lines = f.readlines()
            for a_var_name, a_line in zip(ret.__dict__.keys(), lines):
                a_var = getattr(ret, a_var_name)
                # Below for clarity and extensibility, as we could use directly the class, class(a_line)
                if isinstance(a_var, int):
                    setattr(ret, a_var_name, int(a_line))
                elif isinstance(a_var, float):
                    setattr(ret, a_var_name, float(a_line))
                elif isinstance(a_var, str):
                    setattr(ret, a_var_name, a_line.strip())
        # From legacy macro, there is a typo "od_g_range" so the code doesn't do what it should I guess
        # if ret.odrange >= 3:
        #     ret.odgrange = 1.15
        return ret


@dataclasses.dataclass(frozen=False)
class ProjectMeta:
    """
    Class to read and store metadata from a metadata.txt file.
    All fields are explicitly defined with proper type annotations.
    """

    # String fields
    SampleId: str = ""  # e.g. apero2023_tha_bioness_sup2000_017_st66_d_n1
    Scanop: str = ""  # e.g. adelaide_perruchon
    Ship: str = ""  # e.g. thalassa
    Scientificprog: str = ""  # e.g. apero
    Date: str = ""  # e.g. 20230704-0503
    CTDref: str = ""  # e.g. apero_bio_ctd_017
    Otherref: str = ""  # e.g. apero_bio_uvp6_017u
    Nettype: str = ""  # e.g. bioness
    Observation: str = ""  # e.g. no
    SubMethod: str = ""  # e.g. motoda
    Sample_comment: str = ""  # e.g. vol_zooprocess_saisi
    barcode: str = ""  # e.g. ape000000147
    FracId: str = ""  # e.g. d2_4_sur_4

    # Integer fields
    StationId: int = -1  # e.g. 66
    Townb: int = -1  # e.g. 1
    Towtype: int = -1  # e.g. 1
    Netmesh: int = -1  # e.g. 2000
    Netsurf: int = -1  # e.g. 1
    Vol: int = -1  # e.g. 357
    Fracmin: int = -1  # e.g. 2000
    Fracsup: int = -1  # e.g. 999999
    Fracnb: int = -1  # e.g. 4
    Code: int = -1  # e.g. 1
    CellPart: int = -1  # e.g. 1
    Replicates: int = -1  # e.g. 1
    VolIni: int = -1  # e.g. 1
    VolPrec: int = -1  # e.g. 1
    vol_qc: int = -1  # e.g. 1
    depth_qc: int = -1  # e.g. 1
    sample_qc: int = -1  # e.g. 1111

    # Float fields
    Depth: float = -1.0  # e.g. 99999
    Zmax: float = -1.0  # e.g. 1008
    Zmin: float = -1.0  # e.g. 820
    net_duration: float = -1.0  # e.g. 20
    ship_speed_knots: float = -1.0  # e.g. 2
    cable_length: float = -1.0  # e.g. 9999
    cable_angle: float = -1.0  # e.g. 99999
    cable_speed: float = -1.0  # e.g. 0
    nb_jar: float = -1.0  # e.g. 1
    Latitude: float = -1.0  # e.g. 51.4322000
    Longitude: float = -1.0  # e.g. 18.3108000
    latitude_end: float = -1.0  # e.g. 51.4421000
    longitude_end: float = -1.0  # e.g. 18.3417000

    @classmethod
    def read(cls, path: Path) -> "ProjectMeta":
        """
        Read a metadata.txt file and return a ProjectMeta instance with fields
        populated from the file.
        """
        meta = ProjectMeta()
        with open(path, "r") as strm:
            for line in strm:
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    if hasattr(meta, key):
                        field_type = type(getattr(meta, key))
                        try:
                            if field_type == int:
                                setattr(meta, key, int(value))
                            elif field_type == float:
                                setattr(meta, key, float(value))
                            else:
                                setattr(meta, key, value)
                        except ValueError:
                            # If conversion fails, keep as string
                            setattr(meta, key, value)
                    else:
                        # For any fields not explicitly defined, add them as strings
                        setattr(meta, key, value)

        return meta

    def _relevant_attributes(self):
        """
        Generator that yields relevant attributes for writing to a metadata file.
        Yields tuples of (attr_name, attr_value) for non-default attributes.
        """
        for attr_name in dir(self):
            # Skip special attributes, methods, and class variables
            if (
                attr_name.startswith("__")
                or callable(getattr(self, attr_name))
                or attr_name == "__annotations__"
            ):
                continue

            # Get the attribute value
            attr_value = getattr(self, attr_name)

            # Skip default values for primitive types
            if attr_value == "" or attr_value == -1 or attr_value == -1.0:
                continue

            yield attr_name, attr_value

    def to_dict(self) -> dict:
        """
        Convert the ProjectMeta instance to a dictionary using the _relevant_attributes generator.
        Returns a dictionary with attribute names as keys and their values as values.
        """
        return {
            attr_name: attr_value
            for attr_name, attr_value in self._relevant_attributes()
        }

    def write(self, path: Path) -> None:
        """
        Write the ProjectMeta instance to a metadata.txt file in the same format
        as it is read.
        """
        with open(path, "w") as strm:
            # Write all relevant attributes to the file
            for attr_name, attr_value in self._relevant_attributes():
                strm.write(f"{attr_name}= {attr_value}\n")


class PidFile:
    """
    Class to read and parse .pid files which contain metadata and measurements for samples.
    PID files have a sectioned structure with key-value pairs within each section.
    """

    def __init__(self):
        self.sections: Dict[str, Dict[str, str]] = {}
        self.data_rows: List[Dict[str, str]] = []
        self.header_row: List[str] = []

    @classmethod
    def read(cls, path: Path) -> "PidFile":
        """
        Read a .pid file and return a PidFile instance with sections and data populated.
        """
        pid_file = PidFile()
        current_section = None
        in_data_section = False

        with open(path, "r") as strm:
            for line in strm:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Check if this is a section header
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1]
                    pid_file.sections[current_section] = {}
                    continue

                # Check if this is the start of the data section
                if line == "PID":
                    continue

                # Check if this is the data header row
                if line.startswith("!Item;"):
                    in_data_section = True
                    pid_file.header_row = line[1:].split(";")
                    continue

                # If we're in the data section, parse data rows
                if in_data_section:
                    values = line.split(";")
                    if len(values) == len(pid_file.header_row):
                        row_data = {
                            pid_file.header_row[i]: values[i]
                            for i in range(len(values))
                        }
                        pid_file.data_rows.append(row_data)
                    continue

                # Otherwise, parse key-value pairs in the current section
                if current_section and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    pid_file.sections[current_section][key] = value

        return pid_file

    def get_section(self, section_name: str) -> Optional[Dict[str, str]]:
        """
        Get a section by name.
        Returns None if the section doesn't exist.
        """
        return self.sections.get(section_name)

    def get_value(self, section_name: str, key: str, default: Any = None) -> Any:
        """
        Get a value from a section by key.
        Returns the default value if the section or key doesn't exist.
        """
        section = self.get_section(section_name)
        if section:
            return section.get(key, default)
        return default

    def get_data_rows(self) -> List[Dict[str, str]]:
        """
        Get all data rows.
        """
        return self.data_rows

    def get_header_row(self) -> List[str]:
        """
        Get the header row for the data section.
        """
        return self.header_row

    def write(self, path: Path) -> None:
        """
        Write the PidFile instance to a .pid file in the same format as it is read.
        This preserves all sections, key-value pairs, and data rows.
        """
        with open(path, "w") as strm:
            strm.write("PID\n")  # Standard header
            # Write each section with its key-value pairs
            for section_name, section_data in self.sections.items():
                strm.write(f"[{section_name}]\n")
                for key, value in section_data.items():
                    strm.write(f"{key}={value}\n")
                strm.write("\n")

            # Write the data section if there's any data
            if self.header_row:
                strm.write(f"!{';'.join(self.header_row)}\n")

                # Write each data row
                for row in self.data_rows:
                    values = [row.get(header, "") for header in self.header_row]
                    strm.write(f"{';'.join(values)}\n")


class Measurements:
    """
    Class to read and parse measurement files (TSV format) which contain feature measurements for objects.
    Measurement files have a tab-separated format with a header row and data rows.
    """

    def __init__(self):
        self.header_row: List[str] = []
        self.data_rows: List[Dict[str, Any]] = []

    @classmethod
    def read(
        cls, path: Path, typings: Optional[Dict[str, Type]] = None
    ) -> "Measurements":
        """
        Read a measurement file and return a Measurements instance with data populated.

        Args:
            path: Path to the measurement file
            typings: Dictionary mapping column names to types. If None, all values will be kept as strings.

        Returns:
            A Measurements instance with data populated from the file
        """
        measurements = Measurements()

        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            measurements.header_row = reader.fieldnames if reader.fieldnames else []

            for a_line in reader:
                if not typings:
                    measurements.data_rows.append(a_line)
                    continue
                to_add = {}
                for k, v in a_line.items():
                    if k not in typings:
                        to_add[k] = v
                        continue
                    typing = (
                        float
                        if typings[k] == float or typings[k] == np.float64
                        else typings[k]
                    )
                    try:
                        to_add[k] = typing(v)
                    except ValueError:
                        # Some theoretically int features are stored as floats
                        try:
                            flt = float(v)
                            if int(flt) == flt and typing == int:
                                to_add[k] = typing(flt)
                            else:
                                to_add[k] = flt
                        except ValueError:
                            # If all conversions fail, keep as string
                            to_add[k] = v
                measurements.data_rows.append(to_add)

        return measurements

    def get_data_rows(self) -> List[Dict[str, Any]]:
        """
        Get all data rows.
        """
        return self.data_rows

    def get_header_row(self) -> List[str]:
        """
        Get the header row.
        """
        return self.header_row

    def write(self, path: Path) -> None:
        """
        Write the Measurements instance to a file in the same format as it is read.
        This preserves all data rows and the header row.
        """
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.header_row, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.data_rows)
