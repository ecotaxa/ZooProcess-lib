from pathlib import Path
from typing import Optional


class LogReader:
    def __init__(self, logfile: Path) -> None:
        self.logfile = logfile

    def _find_key_in_file(self, key: str) -> Optional[str]:
        with open(self.logfile) as f:
            for line in f:
                if key in line:
                    return line.split("=")[1].strip()
        return None

    def getBackgroundPattern(self) -> str:
        background = self._find_key_in_file("Background_correct_using")
        length = len("20141003_1144")
        return background[:length]
