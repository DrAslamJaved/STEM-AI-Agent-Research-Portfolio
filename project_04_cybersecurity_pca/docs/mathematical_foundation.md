# Mathematical Foundation

## 1. Data matrix

Let

\[
X \in \mathbb{R}^{n\times p}
\]

contain \(n\) observations and \(p\) numerical cybersecurity features.

Examples include:

- packet count;
- connection duration;
- source byte count;
- destination byte count;
- failed login count;
- connection rate;
- SYN packet count.

## 2. Feature standardization

For training feature \(j\), calculate:

\[
\mu_j=\frac{1}{n}\sum_{i=1}^{n}x_{ij}
\]

and sample standard deviation:

\[
s_j=
\sqrt{
\frac{1}{n-1}
\sum_{i=1}^{n}(x_{ij}-\mu_j)^2
}.
\]

The standardized value is:

\[
z_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
\]

Training values \(\mu_j\) and \(s_j\) must be reused for calibration and test
observations.

No mean or standard deviation may be recalculated using the test set.

## 3. Centered standardized matrix

Let

\[
Z \in \mathbb{R}^{n\times p}
\]

be the standardized matrix. The PCA implementation will defensively center it:

\[
Z_c = Z-\mathbf{1}\overline{z}^{\,T},
\]

where \(\overline{z}\) is the vector of feature means in \(Z\).

For correctly standardized training data, these means should be numerically
close to zero.

## 4. Covariance matrix

The sample covariance matrix is:

\[
C=\frac{1}{n-1}Z_c^TZ_c.
\]

The covariance matrix must satisfy:

\[
C=C^T.
\]

## 5. Positive semidefinite property

For any vector \(a\in\mathbb{R}^p\):

\[
a^TCa
=
\frac{1}{n-1}a^TZ_c^TZ_ca
=
\frac{1}{n-1}\lVert Z_ca\rVert_2^2
\geq 0.
\]

Therefore, \(C\) is positive semidefinite and its exact eigenvalues are
nonnegative.

Very small negative computed eigenvalues may arise from floating-point
round-off. They may be accepted only within a documented numerical tolerance.

## 6. Eigenvalue decomposition

Solve:

\[
Cv_j=\lambda_jv_j.
\]

Because \(C\) is real and symmetric:

- all eigenvalues are real;
- eigenvectors can be chosen orthonormally;
- a symmetric eigenvalue solver should be used.

The eigenvalues are sorted:

\[
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p.
\]

The corresponding eigenvectors are sorted in the same order.

## 7. Explained variance

The explained-variance ratio of component \(j\) is:

\[
r_j=
\frac{\lambda_j}
{\sum_{\ell=1}^{p}\lambda_\ell}.
\]

The cumulative explained variance of the first \(k\) components is:

\[
R_k=\sum_{j=1}^{k}r_j.
\]

The baseline component rule selects the smallest \(k\) satisfying:

\[
R_k\geq0.95.
\]

## 8. Principal-component transformation

Let

\[
V_k=[v_1,v_2,\ldots,v_k].
\]

The principal-component scores are:

\[
T=Z_cV_k.
\]

Thus,

\[
T\in\mathbb{R}^{n\times k}.
\]

## 9. Reconstruction

The reconstruction in standardized feature space is:

\[
\widehat Z_c=TV_k^T.
\]

After adding the PCA centering mean:

\[
\widehat Z=
\widehat Z_c+\mathbf{1}\overline{z}^{\,T}.
\]

## 10. Reconstruction error

For observation \(i\), define:

\[
e_i=
\frac{1}{p}
\left\lVert
z_i-\widehat z_i
\right\rVert_2^2.
\]

Equivalently:

\[
e_i=
\operatorname{mean}
\left[
(z_i-\widehat z_i)^2
\right].
\]

All errors must be calculated in standardized space so that features with large
physical scales do not automatically dominate the anomaly score.

## 11. Anomaly threshold

Using normal calibration errors:

\[
\tau=Q_{0.99}(e_1,e_2,\ldots,e_m).
\]

The prediction rule is:

\[
\widehat y_i=
\begin{cases}
1, & e_i>\tau,\\
0, & e_i\leq\tau.
\end{cases}
\]

## 12. Mathematical validation requirements

The implementation must verify:

1. covariance symmetry;
2. descending eigenvalue order;
3. \(Cv_j\approx\lambda_jv_j\);
4. eigenvector orthonormality;
5. nonnegative eigenvalues within numerical tolerance;
6. explained-variance ratios sum to one for the full eigensystem;
7. cumulative explained variance is monotonic;
8. full-component reconstruction is approximately exact;
9. aggregate reconstruction error does not increase as components are added;
10. agreement with scikit-learn at the principal-subspace level;
11. numerical stability for near-collinear data.

An eigenvector and its negative represent the same direction. Therefore,
eigenvector signs must not be compared directly.

When eigenvalues are repeated, different orthonormal bases may represent the
same principal subspace. Projection matrices should be compared instead.