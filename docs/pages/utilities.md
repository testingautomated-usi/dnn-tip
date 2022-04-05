---
layout: page
title: Utilities
permalink: /utilities/
has_children: false
nav_order: 60
---

## Utilities

### APFD
A common metric to assess test prioritization, implemented as defined by [Feng et. al.](https://dl.acm.org/doi/abs/10.1145/3395363.3397357).

Usage example:

```python
from dnn_tip.apfd import apfd_from_order

# As inputs, you'll need:
# - *is_fault* An 1-D boolean array specifying for each test if it is a fault
# - *index_order* An 1-D integer array specifying assessed order by index
#   (e.g. [2,1,3,0] means that test-2 has highest priority, test-1 has second highest priority, etc.)
apfd_value = apfd_from_order(is_fault, index_order)
```


### CAM (Coverage-Additive Method)

CAM prioritizes tests, given their (neuron or surprise) coverage profiles,
to optimize the overall coverage of the selected tests. 
Note that [our](.paper/) and [previous](https://dl.acm.org/doi/abs/10.1145/3395363.3397357),
research shows hardly any advantage from using CAM, but a large computational overhead.

Usage example:

```python
from dnn_tip import prioritizers

# a coverage "score" is the coverage score for a single test
#   For neuron coverage, it is the number of neurons (or segments) covered by the test.
#   For surprise coverage, it is the surprise adequacy score of the test.

# CAM is implemented as a generator (allowing lazy collection of just a few tests).
#   Hence, to get the full priority order, we need to iterate over the generator.
priority_order = list(prioritizers.cam(scores, profile))
```



