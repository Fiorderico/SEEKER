"""Graph-based equivalence helpers for identical molecular fragments."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .geometry import BondGraph, canonical_element
from .models import Molecule


def connected_components(graph: BondGraph) -> tuple[tuple[int, ...], ...]:
    """Return deterministic connected components of a molecular graph."""

    remaining = set(range(len(graph)))
    result: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[int] = set()
        while stack:
            atom = stack.pop()
            if atom in component:
                continue
            component.add(atom)
            stack.extend(graph[atom] - component)
        remaining -= component
        result.append(tuple(sorted(component)))
    return tuple(result)


def component_isomorphisms(
    left: Molecule,
    left_graph: BondGraph,
    left_atoms: Sequence[int],
    right: Molecule,
    right_graph: BondGraph,
    right_atoms: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate element- and bond-preserving maps from ``left`` to ``right``.

    Every returned tuple follows ``left_atoms`` order and contains the matching
    atom index in ``right``.  Restricting the map to an induced connected
    component makes exchanges of terminal atoms (for example the two hydrogen
    atoms of water) explicit graph automorphisms rather than element-only swaps.
    """

    left_indices = tuple(int(index) for index in left_atoms)
    right_indices = tuple(int(index) for index in right_atoms)
    if len(left_indices) != len(right_indices):
        return ()
    if not left_indices:
        return ((),)

    left_set = set(left_indices)
    right_set = set(right_indices)

    def descriptor(
        molecule: Molecule,
        graph: BondGraph,
        index: int,
        component: set[int],
    ) -> tuple[str, int, tuple[str, ...]]:
        neighbours = tuple(sorted(graph[index] & component))
        return (
            canonical_element(molecule.atoms[index].element),
            len(neighbours),
            tuple(
                sorted(
                    canonical_element(molecule.atoms[item].element)
                    for item in neighbours
                )
            ),
        )

    left_descriptors = {
        index: descriptor(left, left_graph, index, left_set) for index in left_indices
    }
    right_descriptors = {
        index: descriptor(right, right_graph, index, right_set)
        for index in right_indices
    }
    if Counter(left_descriptors.values()) != Counter(right_descriptors.values()):
        return ()

    candidates = {
        index: tuple(
            item
            for item in right_indices
            if right_descriptors[item] == left_descriptors[index]
        )
        for index in left_indices
    }
    order = tuple(
        sorted(
            left_indices,
            key=lambda index: (len(candidates[index]), left_descriptors[index], index),
        )
    )
    mapping: dict[int, int] = {}
    used: set[int] = set()
    results: list[tuple[int, ...]] = []

    def visit(position: int) -> None:
        if position == len(order):
            results.append(tuple(mapping[index] for index in left_indices))
            return
        left_index = order[position]
        for right_index in candidates[left_index]:
            if right_index in used:
                continue
            compatible = True
            for mapped_left, mapped_right in mapping.items():
                left_bonded = mapped_left in left_graph[left_index]
                right_bonded = mapped_right in right_graph[right_index]
                if left_bonded != right_bonded:
                    compatible = False
                    break
            if not compatible:
                continue
            mapping[left_index] = right_index
            used.add(right_index)
            visit(position + 1)
            used.remove(right_index)
            del mapping[left_index]

    visit(0)
    return tuple(sorted(results))


def equivalent_fragment_groups(
    molecule: Molecule,
    graph: BondGraph,
    fragments: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Group fragment positions that are exactly graph-isomorphic."""

    groups: list[list[int]] = []
    for position, fragment in enumerate(fragments):
        for group in groups:
            representative = group[0]
            if component_isomorphisms(
                molecule,
                graph,
                fragments[representative],
                molecule,
                graph,
                fragment,
            ):
                group.append(position)
                break
        else:
            groups.append([position])
    return tuple(tuple(group) for group in groups)
