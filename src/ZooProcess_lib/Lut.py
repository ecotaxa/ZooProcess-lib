# A .lut file contains processing directives for a whole project

from pathlib import Path


class Lut:
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
    def read(path: Path) -> "Lut":
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
        # From legacy macro, there is a typo "od_g_range" so the code doesn't do what it should I guess
        # if ret.odrange >= 3:
        #     ret.odgrange = 1.15
        return ret
