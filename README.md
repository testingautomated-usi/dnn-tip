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

[//]: # (TODO LINK)

If you want to reproduce our exact experiments, there's a reproduction package and docker stuff available at **TODO LINK**.




## Installation
It's as easy as `pip install dnn-tip`.




## Documentation

Find the documentation at [https://testingautomated-usi.github.io/dnn-tip/](https://testingautomated-usi.github.io/dnn-tip/).
