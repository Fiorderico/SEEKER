#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Genetic algorithm variant that evaluates energies via PySCF and
applies torsion mutations via quaternion-based rotations.

This script mirrors the multi-objective GA provided in ``ga_gradient_LLM.py``
while removing the LLM-based induction operator and replacing Gaussian calls
with an in-process PySCF Hartree–Fock single-point evaluation. Torsional
alleles are converted to Cartesian coordinates by applying axis-angle
rotations using quaternions so that every evaluation starts from the original
reference geometry.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from pyscf import gto, scf
except ImportError as exc:  # pragma: no cover - dependency availability check
    raise SystemExit(
        "PySCF non è installato. Installa il pacchetto con 'pip install pyscf'."
    ) from exc

# ---------------------------------------------------------------------------
# Configuration parameters (kept aligned with the legacy implementation)
# ---------------------------------------------------------------------------

INPUT_FILE = ""          # reference .gjf
TMP_DIR = "tmp"
GENERATIONS_DIR = ""

NUM_GENERAZIONI = 20
POPOLAZIONE_INIZIALE = 20
POPOLAZIONE_TARGET = 10

BASE_RATE_MUTATION = 0.45
BASE_RATE_CROSSOVER = 0.8
DELTA_RATE_MUTATION = 0.1
DELTA_RATE_CROSSOVER = 0.1
NUM_OSCILLATIONS = 2

HB_SPHERE = 2.5
HB_BONUS_PER_BOND = 0.02
HB_MUTUAL_PENALTY = 0.05

HB_CONTACT_THRESHOLD = -0.3

HB_EPS_BASE = 1.0
HB_EPS_OHN = 1.1
HB_EPS_OHO = 1.2
HB_EPS_NHO = 1.1
HB_EPS_SHO = 1.0
HB_EPS_SHN = 1.0
HB_EPS_OHS = 1.2
HB_EPS_NHS = 1.2

HH_EPS = 0.1
HH_ALPHA = 15.0
HH_XM = 1.5  # Å

SBX_ETA = 15.0
ANGLE_LOW = 0.0
ANGLE_HIGH = 360.0

USE_STANDARD_ORIENTATION = True  # kept for compatibility, unused here

PAIR_SPHERE = 2.5
NEAR_PAIRS: List[Tuple[int, int]] = []
NEAR_WEIGHTS: List[float] = []

GENI: List[Tuple[int, str]] = []

# Mapping symbol -> atomic number and inverse -------------------------------------------------
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}
INV_Z = {v: k for k, v in ELEMENT_Z.items()}

COV_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.01}
VDW_RADII = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80}

OBJECTIVES: List[str] = [
    "fitness_hbond",
    "fitness_hb_bifork",
    "fitness_energy",
    "fitness_grms",
    "fitness_gmax",
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def dist3(pa: Sequence[float], pb: Sequence[float]) -> float:
    dx = pa[0] - pb[0]
    dy = pa[1] - pb[1]
    dz = pa[2] - pb[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def circular_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


_DISCRETE_PROBS_CACHE: Dict[Tuple[int, float], Tuple[List[float], List[float]]] = {}


def generate_random_allele_discrete(periodicity: int, step_degrees: float = 20.0) -> float:
    w = 0.08
    b = 1 / (2 * math.pi) - w
    n = periodicity
    if step_degrees <= 0:
        step_degrees = 10.0
    npts = max(1, int(round(360.0 / step_degrees)))
    step = 360.0 / npts
    angles_deg = [i * step for i in range(npts)]
    cache_key = (n, round(step, 6))
    if cache_key in _DISCRETE_PROBS_CACHE:
        angles_deg_cached, cumprobs = _DISCRETE_PROBS_CACHE[cache_key]
        if len(angles_deg_cached) == len(angles_deg):
            r = random.random()
            for a, cp in zip(angles_deg_cached, cumprobs):
                if r <= cp:
                    return a
            return angles_deg_cached[-1]
    weights = []
    for ang in angles_deg:
        theta = math.radians(ang)
        dens = 1.0 + b + ((-1.0) ** n) * math.cos(n * theta)
        if dens < 1e-12:
            dens = 1e-12
        weights.append(dens)
    s = sum(weights)
    probs = [w_i / s for w_i in weights]
    cumprobs = []
    acc = 0.0
    for p in probs:
        acc += p
        cumprobs.append(acc)
    cumprobs[-1] = 1.0
    _DISCRETE_PROBS_CACHE[cache_key] = (angles_deg, cumprobs)
    r = random.random()
    for a, cp in zip(angles_deg, cumprobs):
        if r <= cp:
            return float(a)
    return float(angles_deg[-1])


def generate_random_allele(periodicity: int) -> float:
    return generate_random_allele_discrete(periodicity)


def resample_random_alleles(geni: Sequence[Tuple[int, str]]) -> List[float]:
    return [generate_random_allele(period) for period, _ in geni]


def tweak_some_alleles_random(
    alleles: Sequence[float],
    geni: Sequence[Tuple[int, str]],
    k: Optional[int] = None,
    eps_deg: float = 1e-6,
) -> Tuple[List[float], List[int]]:
    n = len(alleles)
    if n == 0:
        return list(alleles), []
    if k is None:
        k = random.randint(1, n)
    k = max(1, min(k, n))
    idxs = random.sample(range(n), k)
    new = list(alleles)
    for i in idxs:
        period = geni[i][0]
        old = new[i]
        for _ in range(10):
            cand = generate_random_allele(period)
            if circular_diff_deg(cand, old) > eps_deg:
                break
        new[i] = cand
    return new, idxs


def tweak_one_allele_random(
    alleles: Sequence[float],
    geni: Sequence[Tuple[int, str]],
    eps_deg: float = 1e-6,
) -> Tuple[List[float], Optional[int]]:
    if not alleles:
        return list(alleles), None
    i = random.randrange(len(alleles))
    period = geni[i][0]
    old = alleles[i]
    for _ in range(10):
        cand = generate_random_allele(period)
        if circular_diff_deg(cand, old) > eps_deg:
            break
    new = list(alleles)
    new[i] = cand
    return new, i


# ---------------------------------------------------------------------------
# Geometry parsing and bonding utilities
# ---------------------------------------------------------------------------


def parse_charge_multiplicity(lines: Sequence[str]) -> Tuple[int, int, int]:
    """Return (index_after_line, charge, multiplicity)."""

    idx = 0
    while idx < len(lines) and lines[idx].strip().startswith("%"):
        idx += 1
    while idx < len(lines) and lines[idx].strip().startswith("#"):
        idx += 1
    while idx < len(lines) and lines[idx].strip() != "":
        idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        return idx, 0, 1
    tokens = lines[idx].strip().split()
    if len(tokens) >= 2 and all(re.fullmatch(r"[+-]?\d+", t) for t in tokens[:2]):
        charge = int(tokens[0])
        multiplicity = int(tokens[1])
        return idx + 1, charge, multiplicity
    return idx, 0, 1


def parse_gjf_geometry_and_genes(
    gjf_path: str,
) -> Tuple[int, int, List[Tuple[int, float, float, float]], List[Tuple[int, str]]]:
    with open(gjf_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start_idx, charge, multiplicity = parse_charge_multiplicity(lines)
    coords: List[Tuple[int, float, float, float]] = []
    float_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
    token_re = re.compile(
        rf"^\s*([A-Za-z]+|\d+)\s+({float_re})\s+({float_re})\s+({float_re})"
    )
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "":
            idx += 1
            break
        m = token_re.match(line)
        if not m:
            break
        sym = m.group(1)
        xs, ys, zs = m.groups()[1:]
        if sym.isdigit():
            Z = int(sym)
        else:
            Z = ELEMENT_Z.get(sym.capitalize())
            if Z is None:
                raise ValueError(f"Elemento non riconosciuto: {sym}")
        coords.append((Z, float(xs), float(ys), float(zs)))
        idx += 1
    genes = parse_genes_from_lines(lines[idx:])
    return charge, multiplicity, coords, genes


def parse_genes_from_lines(lines: Sequence[str]) -> List[Tuple[int, str]]:
    genes: List[Tuple[int, str]] = []
    pattern = re.compile(
        r"^\s*GENE[\w\-]*\s*\((?P<inside>[^)]*?)\)\s*=\s*(?P<rhs>.+?)\s*$",
        flags=re.IGNORECASE,
    )
    per_re = re.compile(r"periodicity\s*=\s*(\d+)", flags=re.IGNORECASE)
    for line in lines:
        m = pattern.match(line)
        if not m:
            continue
        inside = m.group("inside")
        rhs = m.group("rhs").strip()
        pm = per_re.search(inside or "")
        if pm and rhs:
            genes.append((int(pm.group(1)), rhs))
    return genes


def cov_bonded(Z1: int, Z2: int, d: float, tol: float = 0.4) -> bool:
    r1 = COV_RADII.get(Z1, 0.75)
    r2 = COV_RADII.get(Z2, 0.75)
    return d <= (r1 + r2 + tol)


def build_bond_graph(
    coords: Sequence[Tuple[int, float, float, float]]
) -> List[Set[int]]:
    """Return an adjacency list representing covalent connectivity.

    Two atoms are considered bonded when their interatomic distance is below the
    sum of the covalent radii (falling back to generic values) plus a small
    tolerance.  The resulting graph is used to propagate torsion rotations to
    the entire "side" of the molecule connected to the rotating atom.
    """

    n = len(coords)
    P = [(x, y, z) for (_, x, y, z) in coords]
    Z = [Z for (Z, _, _, _) in coords]
    bonds: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = dist3(P[i], P[j])
            if cov_bonded(Z[i], Z[j], dij):
                bonds[i].add(j)
                bonds[j].add(i)
    return bonds


# ---------------------------------------------------------------------------
# H-bond detection utilities (verbatim from the legacy implementation)
# ---------------------------------------------------------------------------

ACCEPTOR_ELEMENTS = (7, 8, 16)  # N, O, S


def identify_donors_acceptors(
    initial_coords: Sequence[Tuple[int, float, float, float]],
    initial_bonds: Sequence[Set[int]],
) -> Tuple[List[int], Dict[int, List[int]], List[int]]:
    Z = [Z for (Z, _, _, _) in initial_coords]
    donors: List[int] = []
    donors_H: Dict[int, List[int]] = {}
    acceptors = [i for i, z in enumerate(Z) if z in ACCEPTOR_ELEMENTS]
    for i, z in enumerate(Z):
        if z in ACCEPTOR_ELEMENTS:
            Hs = [j for j in initial_bonds[i] if Z[j] == 1]
            if Hs:
                donors.append(i)
                donors_H[i] = Hs
    return donors, donors_H, acceptors


def graph_distance_leq(
    bonds: Sequence[Set[int]], i: int, j: int, max_hops: int
) -> bool:
    if bonds is None:
        return False
    if i == j:
        return True
    seen = {i}
    dq: deque[Tuple[int, int]] = deque([(i, 0)])
    while dq:
        u, d = dq.popleft()
        if d >= max_hops:
            continue
        for v in bonds[u]:
            if v == j:
                return True
            if v not in seen:
                seen.add(v)
                dq.append((v, d + 1))
    return False


def hbond_pair_energy(xk: float, xm: float, alpha: float = 15.0, eps: float = 1.0) -> float:
    if xm <= 1e-12:
        return 0.0
    t = xk / xm
    return eps * (
        math.exp(alpha * (1.0 - t))
        - (t * t - 2.0 * t + 3.0) * math.exp(0.5 * alpha * (1.0 - t))
    )


def evaluate_bifurcated_hbond_fitness(
    coords: Sequence[Tuple[int, float, float, float]],
    donors: Sequence[int],
    donors_H: Dict[int, Sequence[int]],
    acceptors: Sequence[int],
    contact_threshold: float = HB_CONTACT_THRESHOLD,
    bonds: Optional[Sequence[Set[int]]] = None,
    xm_scale: float = 1.0,
    xm_add: float = 0.1,
    alpha_bif: float = 5.0,
) -> Tuple[float, Dict[str, object]]:
    if not coords:
        return 0.0, {
            "acceptor_counts": {},
            "pairs": [],
            "threshold": contact_threshold,
            "xm_scale": xm_scale,
            "xm_add": xm_add,
            "alpha_bif": alpha_bif,
        }
    P = [(x, y, z) for (_, x, y, z) in coords]
    Z = [Z for (Z, _, _, _) in coords]
    xm_base = HB_SPHERE
    xm_mod = xm_base * xm_scale + xm_add
    alpha = alpha_bif
    hbond_by_acceptor: Dict[int, List[Tuple[int, int, float]]] = {}
    for acc in acceptors:
        hbond_by_acceptor[acc] = []
    for donor in donors:
        donor_H_list = donors_H.get(donor, [])
        for H in donor_H_list:
            for acc in acceptors:
                if acc == donor:
                    continue
                if bonds is not None and graph_distance_leq(bonds, donor, acc, 2):
                    continue
                xk = dist3(P[H], P[acc])
                eps = HB_EPS_BASE
                key = (Z[donor], Z[acc])
                if key == (8, 7):
                    eps = HB_EPS_OHN
                elif key == (8, 8):
                    eps = HB_EPS_OHO
                elif key == (7, 8):
                    eps = HB_EPS_NHO
                elif key == (16, 8):
                    eps = HB_EPS_SHO
                elif key == (16, 7):
                    eps = HB_EPS_SHN
                elif key == (8, 16):
                    eps = HB_EPS_OHS
                elif key == (7, 16):
                    eps = HB_EPS_NHS
                Ek = hbond_pair_energy(xk, xm_mod, alpha=alpha, eps=eps)
                if Ek <= contact_threshold:
                    hbond_by_acceptor.setdefault(acc, []).append((donor, H, Ek))
    counts: Dict[int, int] = {}
    pairs: List[Tuple[int, int, int, float]] = []
    total_score = 0.0
    for acc, lst in hbond_by_acceptor.items():
        counts[acc] = len(lst)
        if len(lst) >= 2:
            Ek_sorted = sorted(lst, key=lambda x: x[2])
            score = -sum(Ek for (_, _, Ek) in Ek_sorted[:2])
            total_score += score
            for donor, H, Ek in Ek_sorted:
                pairs.append((acc, donor, H, Ek))
    return total_score, {
        "acceptor_counts": counts,
        "pairs": pairs,
        "threshold": contact_threshold,
        "xm_scale": xm_scale,
        "xm_add": xm_add,
        "alpha_bif": alpha_bif,
    }


def evaluate_hbond_fitness(
    coords: Sequence[Tuple[int, float, float, float]],
    donors: Sequence[int],
    donors_H: Dict[int, Sequence[int]],
    acceptors: Sequence[int],
    sphere: float = HB_SPHERE,
    bonus: float = HB_BONUS_PER_BOND,
    mutual_penalty: float = HB_MUTUAL_PENALTY,
    bonds: Optional[Sequence[Set[int]]] = None,
) -> Tuple[float, Dict[str, object], bool]:
    if not coords:
        return float("inf"), {}, False
    P = [(x, y, z) for (_, x, y, z) in coords]
    Z = [Z for (Z, _, _, _) in coords]
    hb_sum = 0.0
    mutual_pairs: List[Tuple[int, int]] = []
    detailed_pairs: List[Dict[str, object]] = []
    hh_pen_sum = 0.0
    hh_pairs: List[Tuple[int, int, float]] = []
    helped = False
    for donor in donors:
        donor_H_list = donors_H.get(donor, [])
        for H in donor_H_list:
            for acc in acceptors:
                if acc == donor:
                    continue
                if bonds is not None and graph_distance_leq(bonds, donor, acc, 2):
                    continue
                xk = dist3(P[H], P[acc])
                xm = sphere
                eps = HB_EPS_BASE
                key = (Z[donor], Z[acc])
                if key == (8, 7):
                    eps = HB_EPS_OHN
                elif key == (8, 8):
                    eps = HB_EPS_OHO
                elif key == (7, 8):
                    eps = HB_EPS_NHO
                elif key == (16, 8):
                    eps = HB_EPS_SHO
                elif key == (16, 7):
                    eps = HB_EPS_SHN
                elif key == (8, 16):
                    eps = HB_EPS_OHS
                elif key == (7, 16):
                    eps = HB_EPS_NHS
                Ek = hbond_pair_energy(xk, xm, alpha=HH_ALPHA, eps=eps)
                hb_sum += Ek
                detailed_pairs.append(
                    {
                        "donor": donor,
                        "acceptor": acc,
                        "H": H,
                        "distance": xk,
                        "energy": Ek,
                    }
                )
                if Ek < 0:
                    helped = True
    for i, donor1 in enumerate(donors):
        for donor2 in donors[i + 1 :]:
            if donor1 == donor2:
                continue
            mutual = False
            for H1 in donors_H.get(donor1, []):
                for H2 in donors_H.get(donor2, []):
                    if dist3(P[H1], P[donor2]) < sphere and dist3(P[H2], P[donor1]) < sphere:
                        mutual = True
                        break
                if mutual:
                    break
            if mutual:
                hb_sum += mutual_penalty
                mutual_pairs.append((donor1, donor2))
    all_H = [i for i, z in enumerate(Z) if z == 1]
    for i, a in enumerate(all_H):
        for b in all_H[i + 1 :]:
            d = dist3(P[a], P[b])
            if d < HH_XM:
                hh_pen = HH_EPS * math.exp(HH_ALPHA * (1.0 - d / HH_XM))
                hh_pen_sum += hh_pen
                hh_pairs.append((a, b, hh_pen))
    hb_sum += hh_pen_sum
    if helped:
        hb_sum -= bonus
    details = {
        "pairs": detailed_pairs,
        "mutual": mutual_pairs,
        "hh_penalty": {"sum": hh_pen_sum, "pairs": hh_pairs},
    }
    return hb_sum, details, helped


# ---------------------------------------------------------------------------
# Quaternion-based torsion manipulation
# ---------------------------------------------------------------------------


def parse_dihedral_atoms(definition: str) -> Optional[Tuple[int, int, int, int]]:
    m = re.search(
        r"D\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
        definition,
        re.IGNORECASE,
    )
    if not m:
        return None
    return tuple(int(m.group(i)) - 1 for i in range(1, 5))


@dataclass
class GeneInfo:
    """Container describing how to interpret and apply a torsional gene."""

    period: int
    definition: str
    atoms: Tuple[int, int, int, int]
    rotate_set: Set[int]


def compute_rotation_group(
    bonds: Sequence[Set[int]],
    j_idx: int,
    k_idx: int,
    l_idx: int,
) -> Set[int]:
    """Return atoms that must follow the rotation around the j–k bond.

    The molecular graph is explored with a depth-first search that starts from
    atom ``l_idx`` and never traverses the ``j_idx``–``k_idx`` bond.  This
    effectively "cuts" the graph on the rotation axis so every atom on the
    ``l`` side of the torsion (except the pivot atoms themselves) is rotated.
    """

    n = len(bonds)
    for name, idx in {"j": j_idx, "k": k_idx, "l": l_idx}.items():
        if idx < 0 or idx >= n:
            raise ValueError(
                f"Indice atomo fuori range per la torsione ({name}={idx + 1}, atomi totali={n})."
            )

    rotate: Set[int] = set()
    stack = [l_idx]
    visited: Set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        if idx not in (j_idx, k_idx):
            rotate.add(idx)
        for nb in bonds[idx]:
            if nb < 0 or nb >= n:
                continue
            if (idx == j_idx and nb == k_idx) or (idx == k_idx and nb == j_idx):
                continue
            if nb not in visited:
                stack.append(nb)
    return rotate


def rotation_matrix(axis: Sequence[float], angle_rad: float) -> List[List[float]]:
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ax /= norm
    ay /= norm
    az /= norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    return [
        [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
        [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
        [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
    ]


def apply_rotation(
    positions: List[List[float]],
    indices: Iterable[int],
    pivot: Sequence[float],
    axis: Sequence[float],
    angle_deg: float,
) -> None:
    """Rotate selected atoms around ``axis`` passing through ``pivot``.

    ``indices`` contains the atoms that must be moved as a consequence of the
    torsional allele.  The axis is expressed in Cartesian coordinates and the
    rotation angle is in degrees, matching the allele representation.
    """

    if abs(angle_deg) < 1e-9:
        return
    mat = rotation_matrix(axis, math.radians(angle_deg))
    px, py, pz = pivot
    for idx in indices:
        vx = positions[idx][0] - px
        vy = positions[idx][1] - py
        vz = positions[idx][2] - pz
        rx = mat[0][0] * vx + mat[0][1] * vy + mat[0][2] * vz
        ry = mat[1][0] * vx + mat[1][1] * vy + mat[1][2] * vz
        rz = mat[2][0] * vx + mat[2][1] * vy + mat[2][2] * vz
        positions[idx][0] = rx + px
        positions[idx][1] = ry + py
        positions[idx][2] = rz + pz


def compute_dihedral_deg(
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
) -> float:
    def _sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def _dot(a: Sequence[float], b: Sequence[float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def _norm(v: Sequence[float]) -> float:
        return math.sqrt(_dot(v, v))

    b0 = _sub(p0, p1)
    b1 = _sub(p2, p1)
    b2 = _sub(p3, p2)
    b1_norm = _norm(b1)
    if b1_norm < 1e-12:
        return 0.0
    b1_unit = [b1[0] / b1_norm, b1[1] / b1_norm, b1[2] / b1_norm]
    v = [b0[i] - _dot(b0, b1_unit) * b1_unit[i] for i in range(3)]
    w = [b2[i] - _dot(b2, b1_unit) * b1_unit[i] for i in range(3)]
    x = _dot(v, w)
    y = _dot(_cross(b1_unit, v), w)
    return math.degrees(math.atan2(y, x))


def prepare_gene_infos(
    genes: Sequence[Tuple[int, str]],
    bonds: Sequence[Set[int]],
) -> List[GeneInfo]:
    """Pre-compute structural data required to apply torsional alleles."""

    infos: List[GeneInfo] = []
    for period, definition in genes:
        atoms = parse_dihedral_atoms(definition)
        if atoms is None:
            raise ValueError(f"Gene non supportato: {definition}")
        i, j, k, l = atoms
        rotate_set = compute_rotation_group(bonds, j, k, l)
        infos.append(
            GeneInfo(period=period, definition=definition, atoms=atoms, rotate_set=rotate_set)
        )
    return infos


def apply_genes_to_positions(
    base_positions: Sequence[Sequence[float]],
    gene_infos: Sequence[GeneInfo],
    alleles: Sequence[float],
) -> List[List[float]]:
    positions = [list(p) for p in base_positions]
    for allele, info in zip(alleles, gene_infos):
        a, b, c, d = info.atoms
        p0 = positions[a]
        p1 = positions[b]
        p2 = positions[c]
        p3 = positions[d]
        current = compute_dihedral_deg(p0, p1, p2, p3)
        current_mod = current % 360.0
        target = allele % 360.0
        delta = ((target - current_mod + 180.0) % 360.0) - 180.0
        axis = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]]
        pivot = p1
        apply_rotation(positions, info.rotate_set, pivot, axis, delta)
    return positions


# ---------------------------------------------------------------------------
# PySCF evaluation
# ---------------------------------------------------------------------------


def compute_pyscf_properties(
    coords: Sequence[Tuple[int, float, float, float]],
    charge: int,
    multiplicity: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    atom_lines = []
    for Z, x, y, z in coords:
        sym = INV_Z.get(Z)
        if sym is None:
            raise ValueError(f"Elemento con Z={Z} non supportato per PySCF")
        atom_lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    mol = gto.Mole()
    mol.atom = atom_lines
    mol.basis = "sto-3g"
    mol.unit = "Angstrom"
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.verbose = 0
    try:
        mol.build()
    except Exception:
        return None, None, None
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.UHF(mol)
    mf.conv_tol = 1e-9
    energy = mf.kernel()
    if not mf.converged:
        return None, None, None
    try:
        grad = mf.nuc_grad_method().kernel()
    except Exception:
        grad = None
    grms = None
    gmax = None
    if grad is not None:
        sq_sum = 0.0
        count = 0
        max_norm = 0.0
        for gx, gy, gz in grad:
            norm = math.sqrt(gx * gx + gy * gy + gz * gz)
            if norm > max_norm:
                max_norm = norm
            sq_sum += gx * gx + gy * gy + gz * gz
            count += 3
        if count > 0:
            grms = math.sqrt(sq_sum / count)
            gmax = max_norm
    return float(energy), grms, gmax


# ---------------------------------------------------------------------------
# GA population structures
# ---------------------------------------------------------------------------


def snapshot_for_rescue(ind: Dict[str, object]) -> Dict[str, object]:
    return {
        "alleli": ind["alleli"][:],
        "fitness_hbond": ind.get("fitness_hbond", float("inf")),
        "fitness_hb_bifork": ind.get("fitness_hb_bifork", float("inf")),
        "fitness_energy": ind.get("fitness_energy", float("inf")),
        "fitness_grms": ind.get("fitness_grms", float("inf")),
        "fitness_gmax": ind.get("fitness_gmax", float("inf")),
        "num_atoms": ind.get("num_atoms"),
        "xyz_lines": ind.get("xyz_lines", [])[:],
        "helped": ind.get("helped", False),
        "hb_details": copy.deepcopy(ind.get("hb_details", {})),
    }


def clone_from_rescue(ind: Dict[str, object], rescue_pool: List[Dict[str, object]]) -> bool:
    if not rescue_pool:
        return False
    donor = random.choice(rescue_pool)
    ind["alleli"] = donor["alleli"][:]
    for k in OBJECTIVES:
        ind[k] = donor.get(k, float("inf"))
    ind["num_atoms"] = donor.get("num_atoms")
    ind["xyz_lines"] = donor.get("xyz_lines", [])[:]
    ind["xyz_file"] = None
    ind["helped"] = donor.get("helped", False)
    ind["hb_details"] = copy.deepcopy(donor.get("hb_details", {}))
    return True


# ---------------------------------------------------------------------------
# NSGA-II utilities
# ---------------------------------------------------------------------------


def dominates_generic(
    a: Dict[str, float], b: Dict[str, float], objectives: Sequence[str]
) -> bool:
    a_vals = [a.get(k, float("inf")) for k in objectives]
    b_vals = [b.get(k, float("inf")) for k in objectives]
    if not all(math.isfinite(x) for x in a_vals + b_vals):
        return False
    not_worse = all(av <= bv for av, bv in zip(a_vals, b_vals))
    strictly_better = any(av < bv for av, bv in zip(a_vals, b_vals))
    return not_worse and strictly_better


def fast_non_dominated_sort(
    pop: Sequence[Dict[str, object]], objectives: Sequence[str]
) -> List[List[int]]:
    S: Dict[int, List[int]] = {i: [] for i in range(len(pop))}
    n_dom = [0] * len(pop)
    fronts: List[List[int]] = [[]]
    for i, p in enumerate(pop):
        S[i] = []
        n_dom[i] = 0
        for j, q in enumerate(pop):
            if i == j:
                continue
            if dominates_generic(p, q, objectives):
                S[i].append(j)
            elif dominates_generic(q, p, objectives):
                n_dom[i] += 1
        if n_dom[i] == 0:
            p["rank"] = 0
            fronts[0].append(i)
    f = 0
    while fronts[f]:
        Q: List[int] = []
        for i in fronts[f]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    pop[j]["rank"] = f + 1
                    Q.append(j)
        f += 1
        fronts.append(Q)
    fronts.pop()
    return fronts


def crowding_distance(
    front: Sequence[int], pop: Sequence[Dict[str, object]], objectives: Sequence[str]
) -> None:
    if not front:
        return
    for i in front:
        pop[i]["crowding"] = 0.0
    for obj_key in objectives:
        front_sorted = sorted(front, key=lambda idx: pop[idx][obj_key])
        pop[front_sorted[0]]["crowding"] = float("inf")
        pop[front_sorted[-1]]["crowding"] = float("inf")
        vmin = pop[front_sorted[0]][obj_key]
        vmax = pop[front_sorted[-1]][obj_key]
        if vmax == vmin:
            continue
        for k in range(1, len(front_sorted) - 1):
            prev_val = pop[front_sorted[k - 1]][obj_key]
            next_val = pop[front_sorted[k + 1]][obj_key]
            pop[front_sorted[k]]["crowding"] += (next_val - prev_val) / (vmax - vmin)


def assign_pareto_metrics(pop: Sequence[Dict[str, object]]) -> List[List[int]]:
    for ind in pop:
        ind["rank"] = int(1e9)
        ind["crowding"] = 0.0
    fronts = fast_non_dominated_sort(pop, OBJECTIVES)
    for f in fronts:
        crowding_distance(f, pop, OBJECTIVES)
    return fronts


def selection_nsga2(
    population: List[Dict[str, object]], target_size: int
) -> List[Dict[str, object]]:
    fronts = assign_pareto_metrics(population)
    selected: List[Dict[str, object]] = []
    for f in fronts:
        if len(selected) + len(f) <= target_size:
            selected.extend([population[i] for i in f])
        else:
            rest = [population[i] for i in f]
            rest.sort(key=lambda ind: ind["crowding"], reverse=True)
            selected.extend(rest[: target_size - len(selected)])
            break
    return selected


def pareto_tournament_selection(
    population: Sequence[Dict[str, object]], tournament_size: int = 2
) -> Dict[str, object]:
    competitors = random.sample(population, tournament_size)
    competitors.sort(key=lambda ind: (ind.get("rank", 1e9), -ind.get("crowding", 0.0)))
    return competitors[0]


# ---------------------------------------------------------------------------
# Variation operators (SBX + mutation)
# ---------------------------------------------------------------------------


def sbx_crossover_angles(a1: float, a2: float, eta: float = SBX_ETA) -> float:
    delta = ((a2 - a1 + 540.0) % 360.0) - 180.0
    x1 = 0.0
    x2 = delta
    child_rel = _bounded_sbx(x1, x2, -180.0, 180.0, eta)
    child = a1 + child_rel
    return _wrap360(child)


def _bounded_sbx(x1: float, x2: float, L: float, U: float, eta: float) -> float:
    if x1 > x2:
        x1, x2 = x2, x1
    if abs(x2 - x1) < 1e-12:
        return (x1 + x2) * 0.5
    u = random.random()
    beta = 1.0 + 2.0 * (x1 - L) / (x2 - x1)
    alpha = 2.0 - pow(beta, -(eta + 1.0))
    if u <= 1.0 / alpha:
        betaq = pow(u * alpha, 1.0 / (eta + 1.0))
    else:
        betaq = pow(1.0 / (2.0 - u * alpha), 1.0 / (eta + 1.0))
    child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
    beta = 1.0 + 2.0 * (U - x2) / (x2 - x1)
    alpha = 2.0 - pow(beta, -(eta + 1.0))
    if u <= 1.0 / alpha:
        betaq = pow(u * alpha, 1.0 / (eta + 1.0))
    else:
        betaq = pow(1.0 / (2.0 - u * alpha), 1.0 / (eta + 1.0))
    child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
    return child1 if random.random() < 0.5 else child2


def _wrap360(x: float) -> float:
    x = x % 360.0
    if x < 0:
        x += 360.0
    return x


def crossover_sbx(
    parent1: Dict[str, object], parent2: Dict[str, object], eta: float = SBX_ETA
) -> List[float]:
    alleles = []
    for a1, a2 in zip(parent1["alleli"], parent2["alleli"]):
        alleles.append(sbx_crossover_angles(a1, a2, eta=eta))
    return alleles


def mutate(
    alleles: Sequence[float], mutation_rate: float, geni: Sequence[Tuple[int, str]]
) -> List[float]:
    new_alleles: List[float] = []
    for allele, (period, _) in zip(alleles, geni):
        if random.random() < mutation_rate:
            new_alleles.append(generate_random_allele(period))
        else:
            new_alleles.append(allele)
    return new_alleles


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------


def _invalidate(ind: Dict[str, object]) -> None:
    for k in OBJECTIVES:
        ind[k] = float("inf")
    ind["fitness_hb_bifork"] = float("inf")
    ind["num_atoms"] = None
    ind["xyz_lines"] = []
    ind["xyz_file"] = None


def write_xyz_file(
    directory: str,
    individual_id: int,
    hb: float,
    energy: float,
    grms: float,
    gmax: float,
    rank: Optional[int],
    num_atoms: int,
    xyz_lines: Sequence[str],
    hb_bif: Optional[float] = None,
) -> str:
    filename = os.path.join(directory, f"individuo_{individual_id}.xyz")
    with open(filename, "w") as f:
        f.write(f"{num_atoms}\n")
        meta = f"HB={hb}  E={energy}  GRMS={grms}  GMAX={gmax}  Rank={rank}"
        if hb_bif is not None and math.isfinite(hb_bif):
            meta += f"  HBbif={hb_bif}"
        f.write(meta + "\n")
        for line in xyz_lines:
            f.write(line + "\n")
    return filename


def positions_to_xyz_lines(
    Z: Sequence[int], positions: Sequence[Sequence[float]]
) -> List[str]:
    lines: List[str] = []
    for Z_i, (x, y, z) in zip(Z, positions):
        sym = INV_Z.get(Z_i, str(Z_i))
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    return lines


def evaluate_individual(
    ind: Dict[str, object],
    gene_infos: Sequence[GeneInfo],
    base_Z: Sequence[int],
    base_positions: Sequence[Sequence[float]],
    charge: int,
    multiplicity: int,
    initial_bonds: Sequence[Set[int]],
    donors: Sequence[int],
    donors_H: Dict[int, Sequence[int]],
    acceptors: Sequence[int],
    hb_sphere: float = HB_SPHERE,
    hb_bonus: float = HB_BONUS_PER_BOND,
    hb_mutual_penalty: float = HB_MUTUAL_PENALTY,
    max_topology_tries: int = 3,
    rescue_pool: Optional[List[Dict[str, object]]] = None,
) -> None:
    tries = 0
    while True:
        tries += 1
        positions = apply_genes_to_positions(base_positions, gene_infos, ind["alleli"])
        coords = [
            (base_Z[i], positions[i][0], positions[i][1], positions[i][2])
            for i in range(len(base_Z))
        ]
        energy, grms, gmax = compute_pyscf_properties(coords, charge, multiplicity)
        if (
            energy is None
            or grms is None
            or gmax is None
            or not math.isfinite(energy)
            or not math.isfinite(grms)
            or not math.isfinite(gmax)
        ):
            if rescue_pool is not None and tries >= max_topology_tries:
                if clone_from_rescue(ind, rescue_pool):
                    return
            if tries >= max_topology_tries:
                _invalidate(ind)
                return
            ind["alleli"], _ = tweak_some_alleles_random(ind["alleli"], GENI)
            continue
        hb_fit, hb_details, helped = evaluate_hbond_fitness(
            coords,
            donors,
            donors_H,
            acceptors,
            sphere=hb_sphere,
            bonus=hb_bonus,
            mutual_penalty=hb_mutual_penalty,
            bonds=initial_bonds,
        )
        hb_bif_fit, hb_bif_det = evaluate_bifurcated_hbond_fitness(
            coords,
            donors,
            donors_H,
            acceptors,
            contact_threshold=HB_CONTACT_THRESHOLD,
            bonds=initial_bonds,
        )
        xyz_lines = positions_to_xyz_lines(base_Z, positions)
        ind["fitness_energy"] = energy
        ind["fitness_grms"] = grms
        ind["fitness_gmax"] = gmax
        ind["fitness_hbond"] = hb_fit
        ind["fitness_hb_bifork"] = hb_bif_fit
        ind["hb_details"] = {"standard": hb_details, "bifork": hb_bif_det}
        ind["helped"] = helped
        ind["num_atoms"] = len(coords)
        ind["xyz_lines"] = xyz_lines
        ind["xyz_file"] = None
        if rescue_pool is not None and all(
            math.isfinite(ind.get(k, float("inf"))) for k in OBJECTIVES
        ):
            rescue_pool.append(snapshot_for_rescue(ind))
        return


# ---------------------------------------------------------------------------
# Population initialisation and CLI helpers
# ---------------------------------------------------------------------------


def initialize_population(
    pop_size: int, geni: Sequence[Tuple[int, str]]
) -> List[Dict[str, object]]:
    population: List[Dict[str, object]] = []
    for i in range(pop_size):
        alleli = [generate_random_allele(period) for period, _ in geni]
        population.append(
            {
                "id": i,
                "alleli": alleli,
                "fitness_hbond": float("inf"),
                "fitness_hb_bifork": float("inf"),
                "fitness_energy": float("inf"),
                "fitness_grms": float("inf"),
                "fitness_gmax": float("inf"),
                "rank": None,
                "crowding": 0.0,
                "xyz_file": None,
                "num_atoms": None,
                "xyz_lines": [],
                "helped": False,
                "hb_details": {},
            }
        )
    return population


def parse_pairs_arg(pairs_str: str) -> List[Tuple[int, int]]:
    if not pairs_str:
        return []
    items = re.split(r"\s*,\s*", pairs_str.strip())
    pairs = []
    for it in items:
        m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", it)
        if not m:
            raise ValueError(
                f"Formato coppia non valido: '{it}'. Usa es. '1-5,2-7'"
            )
        i, j = int(m.group(1)), int(m.group(2))
        pairs.append((i, j))
    return pairs


def parse_weights_arg(weights_str: str) -> List[float]:
    if not weights_str:
        return []
    items = re.split(r"\s*,\s*", weights_str.strip())
    weights = []
    for it in items:
        weights.append(float(it))
    return weights


def save_statistics(
    file_name: str, generations_data: Sequence[Dict[str, object]]
) -> None:
    fields = [
        "Generation",
        "Avg. HB",
        "MAX HB",
        "MIN HB",
        "Avg. HB Bif.",
        "MAX HB Bif.",
        "MIN HB Bif.",
        "Avg. Energy",
        "MAX Energy",
        "MIN Energy",
        "Avg. GRMS",
        "MAX GRMS",
        "MIN GRMS",
        "Avg. GMAX",
        "MAX GMAX",
        "MIN GMAX",
        "Mutation rate",
        "Crossover rate",
    ]
    with open(file_name, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in generations_data:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    global INPUT_FILE, TMP_DIR, GENERATIONS_DIR
    global NUM_GENERAZIONI, POPOLAZIONE_INIZIALE, POPOLAZIONE_TARGET
    global HB_SPHERE, HB_BONUS_PER_BOND, HB_MUTUAL_PENALTY, HB_CONTACT_THRESHOLD
    global GENI, NEAR_PAIRS, NEAR_WEIGHTS

    parser = argparse.ArgumentParser(
        description="GA basata su PySCF e torsioni quaternioniche"
    )
    parser.add_argument("input_file", type=str, help="File .gjf di riferimento")
    parser.add_argument("--tmp-dir", type=str, default=TMP_DIR, help="Directory temporanea")
    parser.add_argument("--out-dir", type=str, default=GENERATIONS_DIR, help="Directory output")
    parser.add_argument("--num-generazioni", type=int, default=NUM_GENERAZIONI)
    parser.add_argument("--pop-iniziale", type=int, default=POPOLAZIONE_INIZIALE)
    parser.add_argument("--pop-target", type=int, default=POPOLAZIONE_TARGET)
    parser.add_argument("--cpu-fraction", type=float, default=1.0)
    parser.add_argument("--hb-sphere", type=float, default=HB_SPHERE)
    parser.add_argument("--hb-bonus", type=float, default=HB_BONUS_PER_BOND)
    parser.add_argument("--hb-mutual", type=float, default=HB_MUTUAL_PENALTY)
    parser.add_argument("--hb-contact", type=float, default=HB_CONTACT_THRESHOLD)
    parser.add_argument("--pairs", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--max-tries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None, help="Seed RNG (default: None).")
    args = parser.parse_args()

    INPUT_FILE = args.input_file
    TMP_DIR = args.tmp_dir
    GENERATIONS_DIR = args.out_dir or "generations"
    NUM_GENERAZIONI = int(args.num_generazioni)
    POPOLAZIONE_INIZIALE = int(args.pop_iniziale)
    POPOLAZIONE_TARGET = int(args.pop_target)
    HB_SPHERE = float(args.hb_sphere)
    HB_BONUS_PER_BOND = float(args.hb_bonus)
    HB_MUTUAL_PENALTY = float(args.hb_mutual)
    HB_CONTACT_THRESHOLD = float(args.hb_contact)
    NEAR_PAIRS = parse_pairs_arg(args.pairs)
    NEAR_WEIGHTS = parse_weights_arg(args.weights)

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(GENERATIONS_DIR, exist_ok=True)

    charge, multiplicity, initial_coords, GENI = parse_gjf_geometry_and_genes(
        INPUT_FILE
    )
    if not GENI:
        raise RuntimeError("Nessun gene trovato nel file di input.")

    initial_bonds = build_bond_graph(initial_coords)
    donors, donors_H, acceptors = identify_donors_acceptors(
        initial_coords, initial_bonds
    )

    base_Z = [Z for (Z, _, _, _) in initial_coords]
    base_positions = [[x, y, z] for (_, x, y, z) in initial_coords]
    gene_infos = prepare_gene_infos(GENI, initial_bonds)

    population = initialize_population(POPOLAZIONE_INIZIALE, GENI)

    generation_log_path = os.path.join(GENERATIONS_DIR, "generation_log.txt")
    generations_data: List[Dict[str, object]] = []

    rescue_pool: List[Dict[str, object]] = []

    for gen in range(NUM_GENERAZIONI):
        phase = (math.pi * NUM_OSCILLATIONS / max(1, NUM_GENERAZIONI - 1)) * gen
        current_mutation_rate = BASE_RATE_MUTATION + DELTA_RATE_MUTATION * math.sin(
            phase
        )
        current_crossover_rate = BASE_RATE_CROSSOVER - DELTA_RATE_CROSSOVER * math.sin(
            phase
        )

        gen_dir = os.path.join(GENERATIONS_DIR, f"population_{gen}")
        os.makedirs(gen_dir, exist_ok=True)

        max_workers = max(
            1,
            int((os.cpu_count() or 1) * max(0.01, min(1.0, args.cpu_fraction))),
        )
        max_workers = min(max_workers, len(population))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_individual,
                    ind,
                    gene_infos,
                    base_Z,
                    base_positions,
                    charge,
                    multiplicity,
                    initial_bonds,
                    donors,
                    donors_H,
                    acceptors,
                    HB_SPHERE,
                    HB_BONUS_PER_BOND,
                    HB_MUTUAL_PENALTY,
                    args.max_tries,
                    rescue_pool,
                )
                for ind in population
            ]
            for future in as_completed(futures):
                future.result()

        selected = selection_nsga2(population, POPOLAZIONE_TARGET)

        HB_list = [
            ind["fitness_hbond"]
            for ind in selected
            if math.isfinite(ind["fitness_hbond"])
        ]
        HB_BIF_list = [
            ind["fitness_hb_bifork"]
            for ind in selected
            if math.isfinite(ind["fitness_hb_bifork"])
        ]
        E_list = [
            ind["fitness_energy"]
            for ind in selected
            if math.isfinite(ind["fitness_energy"])
        ]
        GRMS_list = [
            ind["fitness_grms"]
            for ind in selected
            if math.isfinite(ind["fitness_grms"])
        ]
        GMAX_list = [
            ind["fitness_gmax"]
            for ind in selected
            if math.isfinite(ind["fitness_gmax"])
        ]

        def _stats(values: Sequence[float]) -> Tuple[
            Optional[float], Optional[float], Optional[float]
        ]:
            if not values:
                return (None, None, None)
            return (sum(values) / len(values), max(values), min(values))

        avg_HB, max_HB, min_HB = _stats(HB_list)
        avg_HB_BIF, max_HB_BIF, min_HB_BIF = _stats(HB_BIF_list)
        avg_E, max_E, min_E = _stats(E_list)
        avg_G, max_G, min_G = _stats(GRMS_list)
        avg_M, max_M, min_M = _stats(GMAX_list)

        with open(generation_log_path, "a") as logf:
            logf.write(f"Generazione {gen}:\n")
            logf.write(
                f"  HB      -> avg: {avg_HB} | min: {min_HB} | max: {max_HB}\n"
            )
            logf.write(
                f"  HBbif   -> avg: {avg_HB_BIF} | min: {min_HB_BIF} | max: {max_HB_BIF}\n"
            )
            logf.write(
                f"  Energy  -> avg: {avg_E}  | min: {min_E}  | max: {max_E}\n"
            )
            logf.write(
                f"  GRMS    -> avg: {avg_G}  | min: {min_G}  | max: {max_G}\n"
            )
            logf.write(
                f"  GMAX    -> avg: {avg_M}  | min: {min_M}  | max: {max_M}\n"
            )
            logf.write(f"  Mutation rate: {current_mutation_rate}\n")
            logf.write(f"  Crossover rate: {current_crossover_rate}\n")
            logf.write("  Individui target (rank, crowding, HB, HBbif, E, GRMS, GMAX):\n")
            for ind in selected:
                extra_bif = (
                    f" | HBbif={ind['fitness_hb_bifork']}"
                    if "fitness_hb_bifork" in ind
                    else ""
                )
                logf.write(
                    f"    id={ind['id']} | rank={ind.get('rank')} | crowd={ind.get('crowding')}"
                    f" | HB={ind['fitness_hbond']}{extra_bif} | E={ind['fitness_energy']}"
                    f" | GRMS={ind['fitness_grms']} | GMAX={ind['fitness_gmax']}"
                )
                if ind.get("helped"):
                    logf.write(" [HBOND]\n")
                else:
                    logf.write("\n")
            logf.write("\n")

        generations_data.append(
            {
                "Generation": gen,
                "Avg. HB": avg_HB,
                "MAX HB": max_HB,
                "MIN HB": min_HB,
                "Avg. HB Bif.": avg_HB_BIF,
                "MAX HB Bif.": max_HB_BIF,
                "MIN HB Bif.": min_HB_BIF,
                "Avg. Energy": avg_E,
                "MAX Energy": max_E,
                "MIN Energy": min_E,
                "Avg. GRMS": avg_G,
                "MAX GRMS": max_G,
                "MIN GRMS": min_G,
                "Avg. GMAX": avg_M,
                "MAX GMAX": max_M,
                "MIN GMAX": min_G,
                "Mutation rate": current_mutation_rate,
                "Crossover rate": current_crossover_rate,
            }
        )

        for ind in selected:
            if ind.get("xyz_lines"):
                n_atoms = ind.get("num_atoms") or len(ind["xyz_lines"])
                ind["xyz_file"] = write_xyz_file(
                    gen_dir,
                    ind["id"],
                    ind["fitness_hbond"],
                    ind["fitness_energy"],
                    ind["fitness_grms"],
                    ind["fitness_gmax"],
                    ind.get("rank"),
                    n_atoms,
                    ind["xyz_lines"],
                    hb_bif=ind.get("fitness_hb_bifork"),
                )

        assign_pareto_metrics(selected)

        new_population: List[Dict[str, object]] = []
        next_id = 0
        while len(new_population) < POPOLAZIONE_INIZIALE:
            parent1 = pareto_tournament_selection(selected, tournament_size=2)
            parent2 = pareto_tournament_selection(selected, tournament_size=2)
            if random.random() < current_crossover_rate:
                child_alleles = crossover_sbx(parent1, parent2, eta=SBX_ETA)
            else:
                child_alleles = parent1["alleli"][:]
            child_alleles = mutate(child_alleles, current_mutation_rate, GENI)
            new_population.append(
                {
                    "id": next_id,
                    "alleli": child_alleles,
                    "fitness_hbond": float("inf"),
                    "fitness_hb_bifork": float("inf"),
                    "fitness_energy": float("inf"),
                    "fitness_grms": float("inf"),
                    "fitness_gmax": float("inf"),
                    "rank": None,
                    "crowding": 0.0,
                    "xyz_file": None,
                    "num_atoms": None,
                    "xyz_lines": [],
                    "helped": False,
                    "hb_details": {},
                }
            )
            next_id += 1

        population = new_population

    save_statistics(os.path.join(GENERATIONS_DIR, "evolution.csv"), generations_data)


if __name__ == "__main__":
    main()

