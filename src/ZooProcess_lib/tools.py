
import os
import sys
from pathlib import Path


def nameit(func):
    from functools import wraps
    @wraps(func)
    def nameit_wrapper(*args, **kwargs):
        print(f'Running: {func.__name__}')
        result = func(*args, **kwargs)
        return result
    return nameit_wrapper



def timeit(func):
    """
    Decorator to calculate the time take by a function
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
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__!r}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def create_folder(path: Path):
    print("create folder:" , path.as_posix())
    p = Path(path)
    try :
        if not os.path.isdir(path):
            #os.mkdir(path)
            #os.makedirs(path, exist_ok=True)
            p.mkdir(parents=True, exist_ok=True)
    except OSError as error: 
        path_str = str(p.absolute)

        eprint("cannot create folder: ", path_str ,", ", str(error))
               
def is_file_exist(path):
    return os.path.exists(path)


