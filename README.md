# DNN-TIP: Common Test Input Prioritizers Library 

[![test](https://github.com/testingautomated-usi/dnn-tip/actions/workflows/test.yml/badge.svg)](https://github.com/testingautomated-usi/dnn-tip/actions/workflows/test.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![docstr-coverage](https://img.shields.io/endpoint?url=https://jsonbin.org/MiWeiss/dnn-tip/badges/docstr-cov)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Python Version](https://img.shields.io/pypi/pyversions/dnn-tip)](https://img.shields.io/pypi/pyversions/dnn-tip)
[![PyPi Deployment](https://badgen.net/pypi/v/dnn-tip?cache=30)](https://pypi.org/project/dnn-tip/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/testingautomated-usi/dnn-tip/blob/develop/LICENSE)
[![DOI](https://zenodo.org/badge/478142616.svg)](https://zenodo.org/badge/latestdoi/478142616)


## Implemented Approaches
* __Surprise Adequacies__
    * Distance-based Surprise Adequacy (DSA)
    * Likelihood-based Surprise Adequacy (LSA)
    * MultiModal-Likelihood-based Surprise Adequacy (MLSA)
    * Mahalanobis-based Surprise Adequacy (MDSA)
    * _abstract_ MultiModal Surprise Adequacy
* __Surprise Coverage__
  * Neuron-Activation Coverage (NAC)
  * K-Multisection Neuron Coverage (KMNC)
  * Neuron Boundary Coverage (NBC)
  * Strong Neuron Activation Coverage (SNAC)
  * Top-k Neuron Coverage (TKNC)
* __Utilities__
    * APFD calculation
    * Coverage-Added and Coverage-Total Prioritization Methods (CAM and CTM)

If you are looking for the uncertainty metrics we also tested (including DeepGini),
head over to the sister repository [uncertainty-wizard](https://github.com/testingautomated-usi/uncertainty-wizard).

If you want to reproduce our exact experiments, there's a reproduction package and docker stuff available at [testingautomated-usi/simple-tip](https://github.com/testingautomated-usi/simple-tip).




## Installation
It's as easy as `pip install dnn-tip`.




## Documentation

Find the documentation at [https://testingautomated-usi.github.io/dnn-tip/](https://testingautomated-usi.github.io/dnn-tip/).


## Citation

Here's the reference to the paper as part of which this library was release:

```
@inproceedings{10.1145/3533767.3534375,
author = {Weiss, Michael and Tonella, Paolo},
title = {Simple Techniques Work Surprisingly Well for Neural Network Test Prioritization and Active Learning (Replicability Study)},
year = {2022},
isbn = {9781450393799},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3533767.3534375},
doi = {10.1145/3533767.3534375},
booktitle = {Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis},
pages = {139–150},
numpages = {12},
keywords = {neural networks, Test prioritization, uncertainty quantification},
location = {Virtual, South Korea},
series = {ISSTA 2022}
}
