# Critical Agent Reasoning

## Why should a large PCA reconstruction error indicate an anomaly?

PCA learns the linear subspace that minimizes total squared reconstruction
error for its fitting observations.

If PCA is fitted using representative normal traffic, the retained principal
components describe the dominant correlation structure of normal behaviour.

A new observation with a large residual perpendicular to that subspace cannot
be reconstructed accurately. It is therefore inconsistent with the learned
normal structure and becomes an anomaly candidate.

This conclusion is statistical. A large reconstruction error does not prove
that the observation is malicious.

## Under what assumptions can this reasoning fail?

### Nonlinear structure

Normal traffic may follow a curved or nonlinear manifold. Linear PCA may
reconstruct valid observations poorly and produce false positives.

### Poor feature scaling

Features measured on large numerical scales may dominate covariance and
reconstruction error if standardization is missing or incorrectly fitted.

### Unusual but legitimate observations

Maintenance, backups, software updates, research experiments, or rare user
activities may be valid even though they differ from common traffic.

### High-dimensional noise

Numerous noisy or irrelevant features can destabilize covariance estimates and
hide meaningful low-variance directions.

### Inappropriate threshold selection

A threshold that is too low creates excessive false positives. A threshold that
is too high fails to identify attacks.

### Contaminated training data

If attacks are included in the normal fitting set, PCA may incorporate attack
behaviour into the learned normal subspace.

### Concept drift

Normal network behaviour may change after deployment. A detector trained on
older traffic may incorrectly flag newer legitimate behaviour.

### In-subspace attacks

An attack can have low reconstruction error when it varies mainly along
high-variance retained principal directions.

### Categorical encoding artifacts

Poor encoding of protocol, service, or connection-state variables can create
Euclidean distances that lack valid cybersecurity meaning.

## Required agent-evaluation standard

A satisfactory AI agent response must:

1. explain the geometric subspace argument;
2. distinguish anomaly evidence from proof of an attack;
3. state the normal-training assumption;
4. identify leakage and contamination risks;
5. discuss scaling and threshold selection;
6. identify false-positive and false-negative consequences;
7. recommend empirical validation rather than relying only on intuition.
