# Taj Mahal
(WIP)
An outlier detection library for python. Named after the Mahalanobis distance metric.

## Build
Pre-requisites:
- python 3.12
- cmake
- boost
- eigen
- pybind11

Build (ideally in a virtual environment) using the following:
```
pip install --upgrade pip build
python -m build
pip install dist/*.so
```

## Motivation
This library and suporting materials were written for my Data Mining
class. I decided to use the project as an opportunity to learn about
writing c-based extensions for python, and this library is the result
of these two needs.
