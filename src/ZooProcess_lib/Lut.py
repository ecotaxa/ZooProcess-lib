# A .lut file contains processing directives for a whole project

from pathlib import Path


class Lut:

    def __init__(self):
        # TODO: not the same defaults as IJ macro
        self.min: int = 0
        self.max: int = 65536
        self.gamma: float = 1
        self.sens: str = "before"
        self.adjust: str = "yes"
        self.odrange: float = 1.8
        self.ratio: float = 1.15
        self.sizelimit: int = 800
        self.overlap: float = 0.07
        self.medianchoice: str = "no"
        self.medianvalue: int = 1
        self.resolutionreduct: int = 2400

    @staticmethod
    def read(path: Path) -> 'Lut':
        ret = Lut()
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
        return ret
