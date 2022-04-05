---
layout: page
title: Installation
permalink: /install/
has_children: false
nav_order: 10
---


## Installation from pypi

For most applications, the easiest way to install the package is to use pip.

```shell
pip install dnn-tip
```

## Manual installation

If you want to change anything in this library, clone the [repo](https://github.com/testingautomated-usi/dnn-tip),
make your changes and then install it using

```shell
pip install -e .
```

Ideally, you want to also run the tests, which needs additional dependencies.

```shell
pip install -e .[test]
```
You can then run the test suite using

```shell
pytest tests
```

If you added a nice feature, or fixed a bug, please consider opening a pull request.
