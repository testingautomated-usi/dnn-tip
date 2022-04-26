---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: Home
nav_order: 0
---


# DNN-TIP: Common Test Input Prioritizers 

A collection of dnn test input prioritizers often used as benchmarks in recent literature.

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
