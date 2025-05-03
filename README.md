# ZooProcess Library

A Python library meant at reproducing as exactly as possible the behavior of ZooProcess-legacy, a tool for processing zooplankton images.

## Features

- Scanner background processing
- Vignette and ROIs extraction from scanned images
- Image segmentation
- Image features generation

## Installation

You can install the package directly from GitHub, it is advised to use a tag, not main.

```bash
pip install git+https://github.com/ecotaxa/ZooProcess-lib.git@v0.1.0
```

Or clone the repository and install locally:

```bash
git clone https://github.com/ecotaxa/ZooProcess-lib.git@v0.1.0
cd ZooProcess-lib
pip install -e .
```

In requirements.txt it should look like:

'''
ZooProcess-lib@git+https://github.com/ecotaxa/ZooProcess-lib.git@v0.1.0
'''

## Requirements

- Python 3.12 or higher
- Dependencies:
  - numpy
  - opencv-python
  - pillow

## Usage

For complete detailed examples, please see the tests directory, especially test_vignetter.py

## Development

### Running Tests

**Note**: You should install manually 'gdown' package before running tests. It's used to download big example TIFF files from a public GDrive folder.

```bash
cd tests
python -m pytest --ignore UNUSED/
```

## License

See the [LICENSE](LICENSE.txt) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
