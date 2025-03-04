# Streamsight

![logo](assets/streamsight-logo.png)

Streamsight is an offline Reccomender Systems (RecSys) evaluation toolkit that respects a global timeline.
The aim is to partition the data into different windows where data is incrementally released for the programmer
to fit, train and submit predictions. This aims to provide a close simulation of an online setting when evaluating
RecSys algorithms.

![full-flow](assets/full-flow.png)

[![PyPI Latest Release](https://img.shields.io/pypi/v/streamsightv2.svg)](https://pypi.org/project/streamsightv2/)&nbsp;&nbsp;
[![Docs](https://github.com/HiIAmTzeKean/Streamsight/actions/workflows/pages/pages-build-deployment/badge.svg)](https://hiiamtzekean.github.io/Streamsight/)&nbsp;&nbsp;
[![Python version](https://img.shields.io/badge/python-3.12.5-blue)](https://www.python.org/downloads/)

## Table of Contents
- [Streamsight](#streamsight)
  - [Table of Contents](#table-of-contents)
  - [Installation with Github](#installation-with-github)
    - [Installation through poetry](#installation-through-poetry)
    - [Installation through pip](#installation-through-pip)
  - [Installation with PyPI](#installation-with-pypi)
  - [Documentation](#documentation)
  - [Report and Citation](#report-and-citation)


## Installation with Github

The package can be installed quickly with python `poetry` or the traditional `pip`
method. The recommended way of installation would be through `poetry` as it will
help install the dependencies along with the package. We assume that the repository
has already been cloned else you can run the following code on terminal before
continuing.

```shell
git clone https://github.com/suenalaba/streamsightv2
cd streamsightv2
```

### Installation through poetry

The following code assumes that you do not have `poetry` installed yet. If you
using MacOS, you might want to consider installing `poetry` with homebrew instead.

```shell
pip install poetry
# MacOS can consider using brew install poetry
poetry install
```

### Installation through pip

The following code below assumes that you have `pip` installed and is in system
PATH.

```shell
pip install -e .
```

## Installation with PyPI

Alternatively `streamsight` is available on PyPi and can be installed through
either of the commands below. The link to PyPI can be found
[here](https://pypi.org/project/streamsight/).

```shell
# To install via pip
pip install streamsightv2

# To install with streamsightv2 as a dependency
poetry add streamsightv2
```

## Documentation

The documentation can be found [here](https://hiiamtzekean.github.io/Streamsight/)
and [repository](https://github.com/suenalaba/streamsightv2) on Github.

## Citation

If you use this library in any part of your work, please cite the following papers:

```
Ng, T. K. (2024). Streamsight: a toolkit for offline evaluation of recommender systems. Final Year Project (FYP), Nanyang Technological University, Singapore. https://hdl.handle.net/10356/181114
```