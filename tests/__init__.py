import pathlib
# import sys
# import unittest

# testPath = str(pathlib.Path(__file__).parents[0].resolve().as_posix())

import pathlib
import sys
projectPath = str(pathlib.Path(__file__).parents[1].resolve().as_posix())
print(projectPath)
if projectPath not in sys.path:
    sys.path.insert(0, projectPath)
