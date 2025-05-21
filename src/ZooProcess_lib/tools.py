import csv
import os
import sys
import time
from pathlib import Path
from typing import Callable, Any, Tuple, List, Dict

import numpy as np
from numpy import ndarray


def nameit(func):
    from functools import wraps

    @wraps(func)
    def nameit_wrapper(*args, **kwargs):
        print(f"Running: {func.__name__}")
        result = func(*args, **kwargs)
        return result

    return nameit_wrapper


def timeit(func: Callable) -> Callable:
    """
    Decorator to print the time taken by a function
    use with
        @timeit
        def fn2mesure(somesArgs): ...
    or
        fn2mesure = timeit(fn2mesure)
        print(fn2mesure(someArgs))
    """

    from functools import wraps

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        total_time, result = measure_time(func, *args, **kwargs)
        # first item in the args, ie `args[0]` is `self`
        print(
            f"Function {func.__name__!r}{args} {kwargs} Took {total_time:.4f} seconds"
        )
        return result

    return timeit_wrapper


def measure_time(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    return total_time, result


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def create_folder(path: Path):
    print("create folder:", path.as_posix())
    p = Path(path)
    try:
        if not os.path.isdir(path):
            # os.mkdir(path)
            # os.makedirs(path, exist_ok=True)
            p.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        path_str = str(p.absolute)

        eprint("cannot create folder: ", path_str, ", ", str(error))


def parse_csv(file_path: Path) -> List[Dict[str, str]]:
    """Parses a CSV file and returns a list of dictionaries."""

    data: List[Dict[str, str]] = []
    with open(file_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            cleaned_row = {}
            for key, value in row.items():
                cleaned_row[key] = (
                    value.strip() if value is not None else None
                )  # Clean whitespace. Handle empty values
            data.append(cleaned_row)  # Add the cleaned row to the list
    return data


def is_file_exist(path):
    return os.path.exists(path)


def graph_connected_components(graph):
    """
    Finds the connected components of an undirected graph. (straight from Gemini)

    Args:
        graph (dict): A dictionary representing the graph where keys are nodes
                       and values are sets (or lists) of their neighbors.

    Returns:
        list: A list of sets, where each set represents a connected component
              of the graph.
    """
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            component = set()
            stack = [node]
            visited.add(node)
            component.add(node)

            while stack:
                current = stack.pop()
                for neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(component)

    return components


class ColumnTemporarilySet(object):
    """Draw a column in an image and restore it on exit."""

    def __init__(self, image: ndarray, col_x: int, value: int):
        self.image = image
        self.col_x = col_x
        self.value = value

    def __enter__(self):
        self.line_sav = np.copy(self.image[:, self.col_x : self.col_x + 1])
        self.image[:, self.col_x : self.col_x + 1] = self.value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.image[:, self.col_x : self.col_x + 1] = self.line_sav
