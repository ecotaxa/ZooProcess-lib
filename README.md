# ZooProcess Library

A Python library meant at reproducing as exactly as possible the behavior of ZooProcess v8.X, a tool for processing Zooscan generated images.

## Features

- Scanner background and plain images processing
- Background removal from plain image
- Image segmentation, AKA ROIs extraction from scanned images
- ROIs features generation
- Utility classes for legacy filesystem usage

## Installation

You can install the package directly from GitHub. It is advised to use a tag, not main which might be unstable.

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

```
ZooProcess-lib@git+https://github.com/ecotaxa/ZooProcess-lib.git@v0.1.0
```

## Requirements

- Python 3.10 or higher
- Dependencies:
  - numpy
  - scipy
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
