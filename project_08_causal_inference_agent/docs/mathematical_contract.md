# Mathematical contract

For unit $i$, treatment is $T_i\in\{0,1\}$, potential outcomes are $Y_i(1)$
and $Y_i(0)$, and pretreatment covariates are $X_i$.

- ATE: $\mathbb{E}[Y(1)-Y(0)]$
- ATT: $\mathbb{E}[Y(1)-Y(0)\mid T=1]$
- CATE: $\mathbb{E}[Y(1)-Y(0)\mid X=x]$

Every run records consistency, conditional exchangeability, positivity, no interference,
intervention definition, temporal order, and transport assumptions. Identification claims are
always conditional on the approved DAG. Empirical diagnostics cannot prove an untestable
causal assumption.
