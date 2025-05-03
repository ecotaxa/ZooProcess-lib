# ZooProcess Library

A Python library meant at reproducing as exactly as possible the behavior of ZooProcess-legacy, a tool for processing zooplankton images.

## Features

- Scanner background processing
- Vignette and ROIs extraction from scanned images
- Image segmentation
- Image features generation

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/ecotaxa/ZooProcess-lib.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/YOUR_USERNAME/ZooProcess-lib.git
cd ZooProcess-lib
pip install -e .
```

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

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
