# processor-ocr-pero

CLI processor for the SoDUCo project. OCR processing.
Takes a series of JSON files in a given directory, and produces an updated version of these files in another directory.
It will look for specific regions and run text line detection and recognition on them.
Some extra options enable to produce other output formats: PAGE XML and ALTO XML.

## Install and tests

```sh
pipenv install
pipenv run python -m pero-cli -i ./tests/input  -o ./tests/output -f json -f image
```

## Usage

```
usage: __main__.py [-h] -i INPUT_DIR -o OUTPUT_DIR -f {json,alto,page,image} [--pero-config-file PERO_CONFIG_FILE]

PERO OCR command line argument

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
  -f {json,alto,page,image}, --export-format {json,alto,page,image}
  --pero-config-file PERO_CONFIG_FILE
```
