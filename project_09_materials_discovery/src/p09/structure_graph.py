"""Deterministic crystal-graph and structure-descriptor construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ELEMENTS = 118
DEFAULT_CUTOFF_ANGSTROM = 5.0
DEFAULT_MAX_NEIGHBORS = 12


@dataclass(frozen=True)
class CrystalGraph:
    """Directed periodic neighbour graph with atomic-number nodes and distances."""

    atomic_numbers: np.ndarray
    edge_index: np.ndarray
    edge_distance: np.ndarray

    def validate(self) -> None:
        atoms = np.asarray(self.atomic_numbers)
        edges = np.asarray(self.edge_index)
        distances = np.asarray(self.edge_distance)
        if atoms.ndim != 1 or not len(atoms) or np.any(atoms < 1) or np.any(atoms > ELEMENTS):
            raise ValueError("atomic numbers must be a nonempty vector in [1, 118]")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] != len(distances):
            raise ValueError("edge index must have shape (2, n_edges)")
        if (np.any(edges < 0) or np.any(edges >= len(atoms)) or np.any(distances <= 0)
                or not np.all(np.isfinite(distances))):
            raise ValueError("edges must be in range and distances must be positive and finite")


def crystal_graph(structure, cutoff_angstrom: float = DEFAULT_CUTOFF_ANGSTROM,
                  max_neighbors: int = DEFAULT_MAX_NEIGHBORS) -> CrystalGraph:
    """Build a sorted radius graph from a pymatgen-compatible periodic structure.

    Each site retains at most ``max_neighbors`` neighbours ordered by distance,
    site index and periodic-image coordinates. Periodic self-images are allowed
    when their distance is positive.
    """
    if not np.isfinite(cutoff_angstrom) or cutoff_angstrom <= 0:
        raise ValueError("cutoff must be positive and finite")
    if not isinstance(max_neighbors, int) or max_neighbors < 1:
        raise ValueError("max_neighbors must be a positive integer")
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=int)
    candidates = []
    for source, neighbours in enumerate(structure.get_all_neighbors(cutoff_angstrom)):
        local = []
        for neighbour in neighbours:
            target, distance = int(neighbour.index), float(neighbour.nn_distance)
            image = tuple(float(x) for x in getattr(neighbour, "image", (0., 0., 0.)))
            if 0 <= target < len(atomic_numbers) and np.isfinite(distance) and distance > 0:
                local.append((distance, target, image))
        local.sort()
        candidates.extend((source, target, distance) for distance, target, _ in local[:max_neighbors])
    candidates.sort()
    edges = np.asarray([(source, target) for source, target, _ in candidates], dtype=int).T
    if not len(candidates):
        edges = np.empty((2, 0), dtype=int)
    graph = CrystalGraph(atomic_numbers, edges,
                         np.asarray([distance for _, _, distance in candidates], dtype=float))
    graph.validate()
    return graph


def structure_descriptors(structure) -> np.ndarray:
    """Transparent fixed-size baseline: element fractions plus global geometry."""
    atomic_numbers = np.asarray(structure.atomic_numbers, dtype=int)
    if atomic_numbers.ndim != 1 or not len(atomic_numbers) or np.any(atomic_numbers < 1) or np.any(atomic_numbers > ELEMENTS):
        raise ValueError("structure must have atomic numbers in [1, 118]")
    fractions = np.bincount(atomic_numbers, minlength=ELEMENTS + 1)[1:].astype(float) / len(atomic_numbers)
    lattice = structure.lattice
    geometry = np.array([len(atomic_numbers), float(structure.density),
                         float(structure.volume) / len(atomic_numbers), *map(float, lattice.abc),
                         *map(float, lattice.angles), float(np.mean(atomic_numbers)),
                         float(np.std(atomic_numbers))])
    if not np.all(np.isfinite(geometry)) or np.any(geometry[:3] <= 0):
        raise ValueError("structure geometry must be finite and positive")
    return np.r_[fractions, geometry]


def graph_summary(graphs: list[CrystalGraph]) -> dict:
    if not graphs:
        raise ValueError("at least one graph is required")
    for graph in graphs:
        graph.validate()
    nodes = np.array([len(g.atomic_numbers) for g in graphs])
    edges = np.array([g.edge_index.shape[1] for g in graphs])
    return {"graphs": len(graphs), "mean_nodes": float(np.mean(nodes)), "max_nodes": int(np.max(nodes)),
            "mean_directed_edges": float(np.mean(edges)), "max_directed_edges": int(np.max(edges)),
            "graphs_without_edges": int(np.sum(edges == 0))}
