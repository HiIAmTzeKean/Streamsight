# Streamsight

The purpose of this Final Year Project is to design and implement a toolkit for
evaluating Recommendation System (RecSys) which respects the temporal aspect
during the data splitting process and incrementally release data as close
to a live production setting as possible.

## Installation

The package can be installed quickly with python `poetry` or the traditional `pip`
method. The recommended way of installation would be through `poetry` as it will
help install the dependencies along with the package.

### Installation with poetry

The following code assumes that you do not have `poetry` installed yet. If you
using MacOS, you might want to consider installing `poetry` with homebrew instead.

```
pip install poetry
# MacOS can consider using brew install poetry
poetry install
```

### Installation with pip

The following code below assumes that you have `pip` installed and is in system
PATH.

```
pip install -e .
```