# Phase 08: fixed-budget structure-model comparison

## Question

On the structure-input `matbench_dielectric` task, does a periodic message-passing model improve the held-out error of a transparent structure-descriptor random forest under a predeclared CPU budget?

## Inputs and outer evaluation

The official Matbench outer folds remain unchanged. Each run obtains training structures and targets from `get_train_and_val_data(fold)` and obtains held-out structures and targets only for final metric calculation. The two models use precisely the same fold and target arrays. A first run on fold 0 is a **pilot**; only the prespecified five-fold run may support a general comparison statement.

The Phase 07 representation lock is retained: a 5.0 Å periodic radius graph, maximum 12 nearest directed neighbours per site, atomic-number nodes, and distance edges. The descriptor baseline has 129 columns: 118 elemental fractions and 11 global structure features.

## Fixed model budget

The descriptor comparator is a `RandomForestRegressor` with 300 trees and the run seed. The graph model is a native PyTorch CPU message-passing network: atomic-number embedding 32, hidden dimension 64, two distance-aware message-passing steps, 16 Gaussian radial basis functions, global mean pooling, AdamW at learning rate 0.001 and weight decay 0.00001. It trains for exactly 40 epochs in batches of 16. Target mean and standard deviation are calculated from the training portion only.

The graph trainer uses one CPU thread, disables the CPU MKLDNN backend, and enables `torch.use_deterministic_algorithms(True)`. Its message and graph-pooling reductions use fixed Python-level destination order rather than scatter updates. A permitted operation without a deterministic implementation fails rather than producing unrepeatable evidence. These controls are reproducibility measures, not model-tuning decisions.

No hyperparameter may be selected by inspecting an outer test fold. Phase 08 is a model-comparison study, not an active-learning or uncertainty-calibration result.

## Metrics and interpretation

The saved JSON records held-out MAE, RMSE and test count for both models. Report per-fold results and the arithmetic fold mean. Do not claim that a lower pilot-fold error is a materials-discovery improvement. If the graph model wins in the full five-fold fixed comparison, it motivates a later uncertainty and acquisition study; it does not establish a label-efficiency benefit by itself.

## Runtime boundary

PyTorch is an optional project extra, installed with `pip install -e ".[graph]"`. PyTorch Geometric is intentionally not required. The default model runs on CPU only with exactly one PyTorch thread. Do not override this setting in the Phase 08 comparison.
