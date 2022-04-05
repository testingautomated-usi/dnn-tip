---
layout: page
title: Neuron Coverage
permalink: /neuroncoverage/
has_children: false
nav_order: 40
---


## Neuron Coverage



### Implemented Neuron Coverages

`dnn-tip` implements the following neuron coverage approaches.
If you use `dnn_tip`, consider checking out 
the recommendations and details in the papers originally proposing said approaches,
and, besides [our paper](./paper), cite them.

| Abb. | Name                              | Proposing Papers                                                                                                       |
|:-----|:----------------------------------|:-----------------------------------------------------------------------------------------------------------------------|
| NAC  | Neuron Activation Coverage        | [Pei et. al., SOSP 17](https://dl.acm.org/doi/abs/10.1145/3132747.3132785) / [arXiv](https://arxiv.org/abs/1705.06640) | 
| KMNC | k-Multisection Neuron Coverage    | [Ma et. al., ASE 18](https://dl.acm.org/doi/10.1145/3238147.3238202) / [arXiv](https://arxiv.org/abs/1803.07519)       | 
| NBC  | Neuron Boundary Coverage          | [Ma et. al., ASE 18](https://dl.acm.org/doi/10.1145/3238147.3238202) / [arXiv](https://arxiv.org/abs/1803.07519)       |
| SNAC | Strong Neuron Activation Coverage | [Ma et. al., ASE 18](https://dl.acm.org/doi/10.1145/3238147.3238202) / [arXiv](https://arxiv.org/abs/1803.07519)       | 
| TKNC | Top-K Neuron Coverage             | [Ma et. al., ASE 18](https://dl.acm.org/doi/10.1145/3238147.3238202) / [arXiv](https://arxiv.org/abs/1803.07519)       | 


### Usage Example

The implementions of neuron coverages in `dnn-tip` map a observed activations
to a set of *coverage profiles*, i.e., boolean vectors indicating *covered (true)* and *uncovered (false)* 
parts (here, dependent on the approach, a 'part' could e.g. be a neuron or a range of activation for a given neuron).

Usage is deliberately simple, and the following code snippet shows how to use the neuron coverage implementations.

```python
nc = NAC(cov_threshold=0.5)
coverage_profile = nc(activations)
```

All neuron coverage implementations take some approach-specific construction parameters,
(e.g. `cov_threshold=0.5` for NAC above). 
The use of the created instance is then the same for all.

Note that for some neuron coverages, you'll have to infer reasonable
construction parameters from the training data (such as min and max activations in KMNC), 
or are tunable hyperparameters (such as the number of top-k neurons in TKNC).
