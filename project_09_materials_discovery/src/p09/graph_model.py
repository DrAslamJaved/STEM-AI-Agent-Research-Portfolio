"""Small dependency-light periodic message-passing regression model.

The implementation uses PyTorch directly rather than PyTorch Geometric so that
the Phase 08 graph comparison can run on a standard Windows CPU installation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random

import numpy as np

from .structure_graph import CrystalGraph

try:  # Keep the non-graph project commands usable without the optional extra.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - depends on optional install
    torch = None
    nn = None


def torch_available() -> bool:
    """Return whether the optional graph runtime is installed."""
    return torch is not None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "Phase 08 requires PyTorch. Install it with: "
            'python -m pip install -e ".[graph]"'
        )


@dataclass(frozen=True)
class GraphTrainingConfig:
    """Fixed Phase 08 CPU training protocol; no outer-test tuning is allowed."""

    cutoff_angstrom: float = 5.0
    max_neighbors: int = 12
    embedding_dim: int = 32
    hidden_dim: int = 64
    message_passing_steps: int = 2
    radial_basis_size: int = 16
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    # One thread is deliberate: it makes CPU message aggregation reproducible.
    cpu_threads: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


def _batch_graphs(graphs: list[CrystalGraph], device):
    """Concatenate variable-size graphs into one disconnected graph batch."""
    require_torch()
    if not graphs:
        raise ValueError("at least one graph is required")
    node_parts, edge_parts, distance_parts, graph_parts = [], [], [], []
    offset = 0
    for graph_id, graph in enumerate(graphs):
        graph.validate()
        node_count = len(graph.atomic_numbers)
        node_parts.append(torch.as_tensor(graph.atomic_numbers, dtype=torch.long))
        graph_parts.append(torch.full((node_count,), graph_id, dtype=torch.long))
        if graph.edge_index.shape[1]:
            edge_parts.append(torch.as_tensor(graph.edge_index + offset, dtype=torch.long))
            distance_parts.append(torch.as_tensor(graph.edge_distance, dtype=torch.float32))
        offset += node_count
    edges = (torch.cat(edge_parts, dim=1) if edge_parts
             else torch.empty((2, 0), dtype=torch.long))
    distances = (torch.cat(distance_parts) if distance_parts
                 else torch.empty((0,), dtype=torch.float32))
    return tuple(item.to(device) for item in (
        torch.cat(node_parts), edges, distances, torch.cat(graph_parts)
    ))


if nn is not None:
    class PeriodicMessagePassingRegressor(nn.Module):
        """Atom embedding + distance-aware directed message passing + mean pooling."""

        def __init__(self, config: GraphTrainingConfig):
            super().__init__()
            self.config = config
            self.atom_embedding = nn.Embedding(119, config.embedding_dim)
            self.node_projection = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim), nn.SiLU(),
            )
            centers = torch.linspace(0.0, config.cutoff_angstrom, config.radial_basis_size)
            self.register_buffer("radial_centers", centers)
            self.message_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_dim + config.radial_basis_size, config.hidden_dim),
                    nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim),
                ) for _ in range(config.message_passing_steps)
            ])
            self.update_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * config.hidden_dim, config.hidden_dim), nn.SiLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                ) for _ in range(config.message_passing_steps)
            ])
            self.readout = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim), nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )

        def _radial_basis(self, distances):
            width = self.config.cutoff_angstrom / max(self.config.radial_basis_size - 1, 1)
            return torch.exp(-((distances[:, None] - self.radial_centers[None, :]) / width) ** 2)

        @staticmethod
        def _ordered_mean(messages, destination, count):
            """Aggregate in a fixed destination order without scatter reductions.

            CPU scatter/index-add updates were not repeatable in the installed
            runtime even with deterministic-algorithms mode enabled. The graph
            sizes in this fixed CPU study are small enough for this explicit,
            autograd-safe reduction.
            """
            rows = []
            for node in range(count):
                selected = messages[destination == node]
                if len(selected):
                    rows.append(selected.sum(dim=0) / len(selected))
                else:
                    rows.append(torch.zeros_like(messages[0]))
            return torch.stack(rows)

        def forward(self, atomic_numbers, edge_index, edge_distance, graph_index):
            hidden = self.node_projection(self.atom_embedding(atomic_numbers))
            if edge_index.shape[1]:
                source, destination = edge_index
                radial = self._radial_basis(edge_distance)
                for message_layer, update_layer in zip(self.message_layers, self.update_layers):
                    messages = message_layer(torch.cat((hidden[source], radial), dim=1))
                    aggregate = self._ordered_mean(messages, destination, hidden.shape[0])
                    hidden = hidden + update_layer(torch.cat((hidden, aggregate), dim=1))
            graph_count = int(graph_index.max().item()) + 1
            pooled = torch.stack([hidden[graph_index == graph_id].mean(dim=0)
                                  for graph_id in range(graph_count)])
            return self.readout(pooled).squeeze(1)
else:  # pragma: no cover - only reached without an optional package
    PeriodicMessagePassingRegressor = None


def _set_seed(seed: int) -> None:
    require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_predict_graph_regressor(train_graphs: list[CrystalGraph], train_targets,
                                test_graphs: list[CrystalGraph], *, seed: int,
                                config: GraphTrainingConfig) -> np.ndarray:
    """Fit the fixed graph model using train data only and predict held-out graphs."""
    require_torch()
    if PeriodicMessagePassingRegressor is None:
        raise RuntimeError("PyTorch model was unavailable")
    targets = np.asarray(train_targets, dtype=float).reshape(-1)
    if len(train_graphs) != len(targets) or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("train graphs and finite targets must have equal nonzero length")
    if not test_graphs:
        raise ValueError("at least one test graph is required")
    _set_seed(seed)
    if config.cpu_threads != 1:
        raise ValueError("Phase 08 fixes CPU threads at one for reproducibility")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # This may only be set once per process; it remains one after first use.
        pass
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    # Fail loudly rather than silently accepting a nondeterministic operation.
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    target_mean, target_std = float(np.mean(targets)), float(np.std(targets))
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("training targets must have positive finite standard deviation")
    model = PeriodicMessagePassingRegressor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    target_tensor = torch.as_tensor((targets - target_mean) / target_std, dtype=torch.float32)
    ordering = np.arange(len(train_graphs))
    model.train()
    for epoch in range(config.epochs):
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(ordering)
        for start in range(0, len(ordering), config.batch_size):
            batch_indices = ordering[start:start + config.batch_size]
            batch = _batch_graphs([train_graphs[int(i)] for i in batch_indices], device)
            prediction = model(*batch)
            loss = torch.mean((prediction - target_tensor[batch_indices].to(device)) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_graphs), config.batch_size):
            batch = _batch_graphs(test_graphs[start:start + config.batch_size], device)
            predictions.append(model(*batch).cpu().numpy())
    return np.concatenate(predictions) * target_std + target_mean
