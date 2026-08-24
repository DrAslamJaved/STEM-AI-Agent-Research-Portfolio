# Agentic Debugging Session 01 — Statistical Mean

## 1. Objective

Validate an AI-assisted implementation of a statistical mean function using
mathematical specifications, automated testing, deliberate fault injection,
and human review.

The mathematical specification is:

mean(X) = sum(X) / n

where n is the number of observations.

---

## 2. Initial Implementation

The implementation was intended to calculate the arithmetic mean of a
numeric dataset.

The correct computational expression is:

```python
return sum(data) / len(data)