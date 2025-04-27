# Example .sh , replace the arg after -k with a test name (pattern!)
# must run in a proper venv
PYTHONPATH=.:../src:$PYTHONPATH pytest -k test_dev_segmentation -s

