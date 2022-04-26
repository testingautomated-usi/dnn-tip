---
layout: page
title: Surprise Adequacy
permalink: /surprise/
has_children: false
nav_order: 20
---

## Surprise Adequacy

Surprise adequacy aims to detect novel (surprising) inputs by comparing them
to the distributions of activations observed in the training set.
As opposed to neuron coverage, typically, surprise adequacy is computed
on the activations of a single layer.

### Implemented Surprise Adequacies

`dnn-tip` implements the following surprise adequacies.
If you use `dnn_tip`, consider checking out 
the recommendations and details in the papers originally proposing said approaches,
and, besides [our paper](./paper), cite them.

| Abb.       | Name                                                                            | Proposing Paper                                                                                                                        |
|:-----------|:--------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| DSA        | Distance-Based SA                                                               | [Kim et. al., ICSE 2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8812069) / [arXiv](https://arxiv.org/abs/1808.08444) (*) | 
| LSA        | Likelihood-Based SA *(based on Kernel-Density estimation)*                      | [Kim et. al., ICSE 2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8812069) /  [arXiv](https://arxiv.org/abs/1808.08444)    | 
| MDSA       | Mahalanobis-Distance based SA                                                   | [Kim et. al., ESEC/FSE 2020](https://dl.acm.org/doi/abs/10.1145/3368089.3417065) / [arXiv](https://arxiv.org/pdf/2006.00894.pdf)       |
| MLDSA      | Multimodal-LSA *(based on Gaussion-Mixture model)*                              | [Kim et. al., AST 2021](https://ieeexplore.ieee.org/document/9462987)                                                                  | 
| MultiModal | *Generic, abstract composition of multiple SA, e.g. for per-class SA and MMDSA* |                                                                                                                                        | 
| MMDSA      | Multimodal-MDSA                                                                 | [Kim et. al., AST 2021](https://ieeexplore.ieee.org/document/9462987)                                                                  | 

(*) Implementation partially based on [Weiss et. al., ICSE-W 2021](https://arxiv.org/abs/2103.05939).
If you use our implementation of DSA, you should cite that paper as well.

### Usage Example

The basic usage is deliberately kept simple, and should be easy to understand given the following
examples:

```python
# Create a LSA or MDSA instance 
sa = LSA(train_activations)  # Or MDSA(train_activations)
surprises = sa(test_activations)
```
```python
# DSA in addition also requires the predicted labels
sa = DSA(train_activations, train_predictions)
surprises = sa(test_activations, test_predictions)
```

```python
# MLSA requires a specification of the number of components in the Gaussian Mixture Model
sa = MLSA(train_activations, num_components=3)
surprises = sa(test_activations)
```

Example usages of Muli-Modal Surprise Adequacy.

```python
# Per-Class SA (as recommended in classification problems), here as an example for MDSA
sa = MultiModalSA.build_by_class(train_activations, train_predictions, lambda x,y: MDSA(x))
# num_threads is optional, executes modals in parallel
surprises = sa(test_activations, num_threads=4)
```

```python
# Multi-Modal MDSA (using k-means for clustering)
#   Check constructor signature for plenty optional params regarding k-means.
sa = MultiModalSA.build_by_class(train_activations) 
surprises = sa(test_activations, num_threads=4)
```

## Mapping surpise adequacies to surprise profiles

While based on our empirical results we do not recommend it, 
you may somtimes want to use coverage profiles instead of surprise adequacies (see our [paper](./paper) for details).

This can be achieved as follows:

```python
mapper =  SurpriseCoverageMapper(
  buckets, # int. Number of buckets for the coverage profile
  limit, # float. Upper limit for the coverage profile
  overflow_bucket # boolean. Use one bucket for "surprises > limit"
)
profile = mapper.get_coverage_profile(surprises)
```


## Recommendations

### DSA
DSA scales badly to large training sets. Hence, a few recommendations:

#### Use a subset of the training set activations

In a related [paper](https://arxiv.org/abs/2103.05939),
we showed that using a subset of the training set activations
can help to drastically improve runtime of DSA while providing similar results.
You can either do that when collection the activations, or pass set the parameters 
  `subsampling` (expects a float between 0 and 1) and `subsampling_seed` (expects an integer) when creating 
  a `dnn-tip` DSA instance.

#### Use parallel computation.
 `dnn-tip` provides a parallel implementation of DSA, thus parallelizing the computation
  is simple as a user, but you should fine-tune the parameters to meet your systems abilities (RAM, most importantly).
  Set the parameters `badge_size` (default: 10) when creating the DSA instance, and set `num_threads` when calculating DSA
  to a positive integer. Example:
```python
sa = DSA(train_activations, train_predictions, badge_size=10)
surprises = sa(test_activations, test_predictions, num_threads=4)
```
Also, in practice, it can be useful to increase SWAP-memory to defend against crashes due to short peak-loads.

### LSA
The KDE used is not numerically stable. 
Our implementation takes various steps to increase stability, and where not possible attempts 
to fail gracefully (returning surprise `0`). 
Still, in practice, you might be better off using a MDSA, which is faster, stable and shows similar results,
but numerically stable and at higher prediction-time performance.

