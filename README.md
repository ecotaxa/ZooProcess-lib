# ZooProcess Library

A Python library meant at reproducing as exactly as possible the behavior of ZooProcess v8.X, a tool for processing Zooscan generated images.

## Features

- Scanner background and plain images processing
- Background removal from plain image
- Image segmentation, AKA ROIs extraction from scanned images
- ROIs features generation
- Utility classes for legacy filesystem usage

## Installation

You can install the package directly from a GitHub release. It is advised to use a specific version release, not the main branch which might be unstable.

```bash
pip install https://github.com/ecotaxa/ZooProcess-lib/archive/refs/tags/v0.6.0.tar.gz
```

Or download the release tarball and install locally:

```bash
# Download the release
wget https://github.com/ecotaxa/ZooProcess-lib/archive/refs/tags/v0.6.0.tar.gz
# Extract the archive
tar -xzf v0.6.0.tar.gz
# Navigate to the extracted directory
cd ZooProcess-lib-0.6.0
# Install the package
pip install -e .
```

In requirements.txt it should look like:

```
ZooProcess-lib @ https://github.com/ecotaxa/ZooProcess-lib/archive/refs/tags/v0.6.0.tar.gz
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
