#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import shutil
from typing import List, Tuple, Dict, Optional, Iterable, Union

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================
# Regex meta dal commento XYZ
# =========================
COMMENT_E_RE  = re.compile(r"\bE\s*=\s*([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)")
COMMENT_HB_RE = re.compile(r"\bHB\s*=\s*([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)")
COMMENT_A_RE  = re.compile(r"\bA\s*=\s*([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)")
COMMENT_B_RE  = re.compile(r"\bB\s*=\s*([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)")
COMMENT_C_RE  = re.compile(r"\bC\s*=\s*([+\-]?\d+(?:\.\d+)?(?:[EeDd][+\-]?\d+)?)")

def _grab_float(rx, s):
    m = rx.search(s)
    if not m: return None
    return float(m.group(1).replace("D","E").replace("d","E"))

def parse_xyz(path: str) -> Tuple[np.ndarray, List[str], Dict[str, Optional[float]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.rstrip("\n") for l in f]
    if len(lines) < 3:
        raise ValueError(f"XYZ troppo corto: {path}")
    try:
        n = int(lines[0].strip())
    except Exception:
        raise ValueError(f"Prima riga non intera in {path}")
    comment = lines[1] if len(lines) > 1 else ""
    atom_lines = lines[2:2+n]
    if len(atom_lines) < n:
        raise ValueError(f"Righe atomiche < N in {path}")
    coords = []
    for ln in atom_lines:
        toks = ln.split()
        if len(toks) < 4:
            raise ValueError(f"Riga atomica malformata in {path}: {ln}")
        x, y, z = map(float, toks[-3:])
        coords.append([x, y, z])
    meta = {"E": _grab_float(COMMENT_E_RE, comment),
            "HB": _grab_float(COMMENT_HB_RE, comment),
            "A": _grab_float(COMMENT_A_RE, comment),
            "B": _grab_float(COMMENT_B_RE, comment),
            "C": _grab_float(COMMENT_C_RE, comment)}
    return np.array(coords, dtype=float), atom_lines, meta

# =========================
# RMSD (Kabsch)
# =========================
def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    if P.shape != Q.shape:
        return np.inf
    if P.size == 0:
        return 0.0
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.linalg.det(V @ Wt)
    Dfix = np.diag([1.0, 1.0, np.sign(d)])
    U = V @ Dfix @ Wt
    Qr = Qc @ U
    diff2 = ((Pc - Qr) ** 2).sum()
    return float(np.sqrt(diff2 / P.shape[0]))

# =========================
# Distanza combinata (RMSD_norm + L1(ABC_norm))
# =========================
def distance_matrix_rmsd_abc(coords_list: List[np.ndarray],
                             ABC_list: np.ndarray,
                             w_rmsd: float = 1.0,
                             w_abc: float = 1.0) -> Tuple[np.ndarray, MinMaxScaler, float]:
    n = len(coords_list)
    if n != ABC_list.shape[0]:
        raise ValueError("coords_list e ABC_list hanno dimensioni non coerenti.")
    abc_scaler = MinMaxScaler()
    ABCn = abc_scaler.fit_transform(ABC_list.astype(float))
    RMSD = np.zeros((n, n), dtype=float)
    for i in range(n):
        Pi = coords_list[i]
        for j in range(i+1, n):
            Qj = coords_list[j]
            RMSD[i, j] = RMSD[j, i] = kabsch_rmsd(Pi, Qj)
    max_r = float(np.max(RMSD)) if n > 1 else 1.0
    if max_r <= 1e-12:
        max_r = 1.0
    RMSD_norm = RMSD / max_r
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d_r = RMSD_norm[i, j] if w_rmsd > 0 else 0.0
            d_a = np.sum(np.abs(ABCn[i] - ABCn[j])) if w_abc > 0 else 0.0
            Dij = w_rmsd * d_r + w_abc * d_a
            D[i, j] = D[j, i] = float(Dij)
    return D, abc_scaler, max_r

# =========================
# Metriche e util su D
# =========================
def silhouette_precomputed_nonnoise(D: np.ndarray, labels: np.ndarray) -> Optional[float]:
    labs = np.asarray(labels)
    mask = labs != -1
    if mask.sum() > 1 and len(set(labs[mask])) > 1:
        try:
            return float(silhouette_score(D[np.ix_(mask, mask)], labs[mask], metric="precomputed"))
        except Exception:
            return None
    return None

def global_diameter(D: np.ndarray) -> float:
    return float(np.max(D)) if D.size else 0.0

def cluster_relative_extent(D: np.ndarray, labels: np.ndarray) -> float:
    labs = np.asarray(labels)
    uniq = sorted(set(labs) - {-1})
    if not uniq:
        return 0.0
    G = global_diameter(D) + 1e-12
    worst = 0.0
    for c in uniq:
        idx = np.where(labs == c)[0]
        if len(idx) < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        local = float(np.max(sub))
        worst = max(worst, local / G)
    return worst

def kdistance_eps_from_D(D: np.ndarray, k: int) -> float:
    n = D.shape[0]
    k = max(1, int(k))
    kth = []
    for i in range(n):
        row = np.copy(D[i])
        row = np.sort(row)
        row = row[row > 0]
        kth.append(0.0 if row.size == 0 else row[min(k-1, row.size-1)])
    kth = np.array(sorted(kth), dtype=float)
    x = np.arange(len(kth), dtype=float)
    x0, y0 = x[0], kth[0]; xN, yN = x[-1], kth[-1]
    num = np.abs((yN - y0)*x - (xN - x0)*kth + xN*y0 - yN*x0)
    den = np.sqrt((yN - y0)**2 + (xN - x0)**2) + 1e-12
    ei = int(np.argmax(num / den))
    return float(kth[ei])

def dbscan_auto_params_precomputed(D: np.ndarray,
                                   min_samples_grid=(3,4,5,6,8,10),
                                   noise_max: float = 0.6,
                                   extent_penalty_weight: float = 0.5) -> Tuple[float, int, Dict]:
    best = dict(score=-1e9, eps=None, ms=None, labels=None, details=None)
    for ms in min_samples_grid:
        eps = kdistance_eps_from_D(D, ms)
        model = DBSCAN(eps=float(eps), min_samples=int(ms), metric="precomputed").fit(D)
        labels = model.labels_
        sil = silhouette_precomputed_nonnoise(D, labels)
        noise = float(np.mean(labels == -1))
        uniq = len(set(labels) - {-1})
        extent = cluster_relative_extent(D, labels)
        sil_val = (-1.0 if sil is None else sil)
        penalty_noise = max(0.0, noise - float(noise_max))
        reward_k = min(uniq, 10) * 0.05
        score = (0.7 * sil_val) - (1.5 * penalty_noise) - (extent_penalty_weight * extent) + reward_k
        if score > best["score"]:
            best = dict(score=score, eps=float(eps), ms=int(ms), labels=labels,
                        details={"silhouette": (None if sil is None else float(sil)),
                                 "noise_frac": noise, "n_clusters": uniq, "extent": extent})
    if best["eps"] is None:
        eps, ms = kdistance_eps_from_D(D, 5), 5
        model = DBSCAN(eps=float(eps), min_samples=int(ms), metric="precomputed").fit(D)
        labels = model.labels_
        best = dict(score=-1e6, eps=float(eps), ms=int(ms), labels=labels,
                    details={"silhouette": silhouette_precomputed_nonnoise(D, labels),
                             "noise_frac": float(np.mean(labels == -1)),
                             "n_clusters": len(set(labels) - {-1}),
                             "extent": cluster_relative_extent(D, labels)})
    return best["eps"], best["ms"], best

# =========================
# MDS: 2D e 3D (per plot)
# =========================
def mds_embed_2d(D: np.ndarray) -> np.ndarray:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto")
    return mds.fit_transform(D)

def mds_embed_3d(D: np.ndarray) -> np.ndarray:
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=0, normalized_stress="auto")
    return mds.fit_transform(D)

# =========================
# PLOT helpers (accettano 1 o più rappresentanti per cluster)
# =========================
RepDict = Dict[int, Union[int, Iterable[int]]]

def _iter_rep_indices(reps: Optional[RepDict]) -> Iterable[Tuple[int,int]]:
    if not reps: return []
    out = []
    for lab, val in reps.items():
        if isinstance(val, (list, tuple, np.ndarray, set)):
            for i in val:
                out.append((int(lab), int(i)))
        else:
            out.append((int(lab), int(val)))
    return out

def plot_clusters_on_mds_2d(X2: np.ndarray, labels: np.ndarray, out_png: str, title: str,
                            representatives: Optional[RepDict] = None):
    labs = np.asarray(labels)
    uniq = sorted(set(labs))
    cmap = plt.cm.get_cmap("tab20", max(2, len(uniq)))

    fig, ax = plt.subplots(figsize=(7,6), dpi=160)
    color_map = {}
    for i, lab in enumerate(uniq):
        idx = (labs == lab)
        c = "#9e9e9e" if lab == -1 else cmap(i)
        color_map[int(lab)] = c
        mk = "x" if lab == -1 else "o"
        ax.scatter(X2[idx,0], X2[idx,1], s=24, c=[c], marker=mk,
                   edgecolors="white" if lab != -1 else "none", linewidths=0.4,
                   label=("noise (-1)" if lab == -1 else f"cluster {lab}"))
    star_labeled = False
    for lab, idx in _iter_rep_indices(representatives):
        if not (0 <= idx < X2.shape[0]): continue
        x, y = X2[idx, 0], X2[idx, 1]
        col = color_map.get(int(lab), "#000000")
        lbl = "representative (★)" if not star_labeled else None
        ax.scatter([x], [y], s=180, marker="*", c=[col],
                   edgecolors="black", linewidths=0.8, zorder=5, label=lbl)
        star_labeled = True

    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2"); ax.set_title(title)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])
    handles, labels_legend = ax.get_legend_handles_labels()
    bylbl = dict(zip(labels_legend, handles))
    ax.legend(bylbl.values(), bylbl.keys(), loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0.5, fontsize=8)
    plt.savefig(out_png, bbox_inches='tight'); plt.close(fig)

def plot_2d_colored(X2: np.ndarray, vals: np.ndarray, out_png: str, label: str, title: str):
    vals = np.asarray(vals, dtype=float)
    mask = ~np.isnan(vals)
    fig, ax = plt.subplots(figsize=(7,6), dpi=140)
    if (~mask).any():
        ax.scatter(X2[~mask,0], X2[~mask,1], s=18, c="#cccccc", edgecolors="none", label="N/A")
    sc = ax.scatter(X2[mask,0], X2[mask,1], s=24, c=vals[mask], cmap="viridis", edgecolors="none")
    cb = fig.colorbar(sc, ax=ax); cb.set_label(label)
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2"); ax.set_title(title)
    plt.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

def plot_clusters_on_mds_3d(X3: np.ndarray, labels: np.ndarray, out_png: str, title: str,
                            representatives: Optional[RepDict] = None):
    labs = np.asarray(labels)
    uniq = sorted(set(labs))
    cmap = plt.cm.get_cmap("tab20", max(2, len(uniq)))

    fig = plt.figure(figsize=(8,6), dpi=160)
    ax = fig.add_subplot(111, projection='3d')
    color_map = {}
    for i, lab in enumerate(uniq):
        idx = (labs == lab)
        c = "#9e9e9e" if lab == -1 else cmap(i)
        color_map[int(lab)] = c
        mk = "x" if lab == -1 else "o"
        ax.scatter(X3[idx,0], X3[idx,1], X3[idx,2], s=22, c=[c], marker=mk,
                   depthshade=False, edgecolors="white" if lab != -1 else "none",
                   linewidths=0.4, label=("noise (-1)" if lab == -1 else f"cluster {lab}"))
    star_labeled = False
    for lab, idx in _iter_rep_indices(representatives):
        if not (0 <= idx < X3.shape[0]): continue
        x, y, z = X3[idx, 0], X3[idx, 1], X3[idx, 2]
        col = color_map.get(int(lab), "#000000")
        lbl = "representative (★)" if not star_labeled else None
        ax.scatter([x], [y], [z], s=220, marker="*",
                   c=[col], edgecolors="black", linewidths=0.8,
                   depthshade=False, zorder=6, label=lbl)
        star_labeled = True
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2"); ax.set_zlabel("MDS 3")
    ax.set_title(title)
    fig.subplots_adjust(right=0.78)
    handles, labels_legend = ax.get_legend_handles_labels()
    bylbl = dict(zip(labels_legend, handles))
    ax.legend(bylbl.values(), bylbl.keys(), loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              fontsize=8, frameon=True)
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

def plot_3d_colored(X3: np.ndarray, vals: np.ndarray, out_png: str, label: str, title: str):
    vals = np.asarray(vals, dtype=float); mask = ~np.isnan(vals)
    fig = plt.figure(figsize=(8,6), dpi=140)
    ax = fig.add_subplot(111, projection='3d')
    if (~mask).any():
        ax.scatter(X3[~mask,0], X3[~mask,1], X3[~mask,2], s=16, c="#cccccc", depthshade=False, label="N/A")
    sc = ax.scatter(X3[mask,0], X3[mask,1], X3[mask,2], s=22, c=vals[mask], cmap="viridis", depthshade=False)
    cb = fig.colorbar(sc, ax=ax); cb.set_label(label)
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2"); ax.set_zlabel("MDS 3"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

# =========================
# PAM (K-Medoids) su metrica precomputed
# =========================
def _pam_total_cost(D: np.ndarray, medoids_idx: List[int]) -> float:
    if len(medoids_idx) == 0:
        return np.inf
    dist_to_med = np.min(D[:, medoids_idx], axis=1)
    return float(np.sum(dist_to_med))

def _pam_build(D: np.ndarray, k: int) -> List[int]:
    n = D.shape[0]
    sums = D.sum(axis=1)
    medoids = [int(np.argmin(sums))]
    while len(medoids) < k:
        best_gain = None
        best_cand = None
        dist_curr = np.min(D[:, medoids], axis=1)
        cost_curr = float(np.sum(dist_curr))
        for h in range(n):
            if h in medoids:
                continue
            dist_new = np.minimum(dist_curr, D[:, h])
            cost_new = float(np.sum(dist_new))
            gain = cost_curr - cost_new
            if (best_gain is None) or (gain > best_gain):
                best_gain = gain
                best_cand = h
        if best_cand is None:
            best_cand = int(np.random.choice([i for i in range(n) if i not in medoids]))
        medoids.append(int(best_cand))
    return medoids

def _pam_swap(D: np.ndarray, medoids: List[int]) -> List[int]:
    n = D.shape[0]
    medoids = medoids[:]
    improved = True
    while improved:
        improved = False
        dist_curr = np.min(D[:, medoids], axis=1)
        cost_curr = float(np.sum(dist_curr))
        med_set = set(medoids)
        for mi, m in enumerate(list(medoids)):
            for h in range(n):
                if h in med_set:
                    continue
                new_medoids = medoids[:]
                new_medoids[mi] = h
                dist_new = np.min(D[:, new_medoids], axis=1)
                cost_new = float(np.sum(dist_new))
                if cost_new + 1e-12 < cost_curr:
                    medoids = new_medoids
                    improved = True
                    break
            if improved:
                break
    return medoids

def pam_kmedoids(D: np.ndarray, k: int) -> Tuple[np.ndarray, List[int]]:
    m = D.shape[0]
    if k < 1 or m == 0:
        return np.zeros(m, dtype=int), []
    if k >= m:
        labels = np.arange(m, dtype=int)
        medoids = list(range(m))
        return labels, medoids
    medoids = _pam_build(D, k)
    medoids = _pam_swap(D, medoids)
    dist_to_med = D[:, medoids]  # m x k
    labels = np.argmin(dist_to_med, axis=1).astype(int)
    return labels, medoids

# =========================
# Representative pickers (singoli)
# =========================
def pick_cluster_candidates_minE(energies: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    reps: Dict[int, int] = {}
    labs = np.asarray(labels)
    for lab in sorted(set(labs) - {-1}):
        idxs = np.where(labs == lab)[0]
        if len(idxs) == 0: continue
        reps[int(lab)] = int(idxs[np.argmin(energies[idxs])])
    return reps

def pick_cluster_medoids(D: np.ndarray, labels: np.ndarray) -> Dict[int,int]:
    reps = {}
    labs = np.asarray(labels)
    for c in sorted(set(labs) - {-1}):
        idx = np.where(labs == c)[0]
        if len(idx) == 0: continue
        sub = D[np.ix_(idx, idx)]
        m_local = int(idx[np.argmin(sub.sum(axis=1))])
        reps[int(c)] = m_local
    return reps

def pick_cluster_minHB(hb_vals: np.ndarray, labels: np.ndarray) -> Dict[int,int]:
    reps = {}
    labs = np.asarray(labels)
    hb = np.asarray(hb_vals, dtype=float)
    for c in sorted(set(labs) - {-1}):
        idx = np.where(labs == c)[0]
        if len(idx) == 0: continue
        local = hb[idx]
        mask = ~np.isnan(local)
        if not mask.any():
            continue
        best_local = idx[np.argmin(local[mask])]
        reps[int(c)] = int(best_local)
    return reps

def pick_cluster_mds_center(X2: np.ndarray, labels: np.ndarray) -> Dict[int,int]:
    reps = {}
    labs = np.asarray(labels)
    for c in sorted(set(labs) - {-1}):
        idx = np.where(labs == c)[0]
        if len(idx) == 0: continue
        P = X2[idx]
        centroid = P.mean(axis=0, keepdims=True)
        i_best = idx[np.argmin(np.linalg.norm(P - centroid, axis=1))]
        reps[int(c)] = int(i_best)
    return reps

def pick_cluster_densest_core(D: np.ndarray, labels: np.ndarray, core_sample_indices: np.ndarray, eps: float) -> Dict[int,int]:
    core_set = set(core_sample_indices.tolist())
    reps = {}
    labs = np.asarray(labels)
    for c in sorted(set(labs) - {-1}):
        idx = np.where(labs == c)[0]
        cores = [i for i in idx if i in core_set] or idx.tolist()
        best_i, best_deg = None, -1
        for i in cores:
            deg = int(np.sum(D[i, idx] <= eps))
            if deg > best_deg:
                best_deg, best_i = deg, i
        reps[int(c)] = int(best_i)
    return reps

# =========================
# Utility per combinare più rappresentanti e deduplicare
# =========================
def unify_representatives(n: int, reps_list: List[Dict[int,int]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for reps in reps_list:
        for lab, idx in reps.items():
            if not (0 <= idx < n):
                continue
            if lab not in out:
                out[lab] = []
            if idx not in out[lab]:
                out[lab].append(int(idx))
    return out

# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="DBSCAN su distanza RMSD_norm + L1(ABC_norm); split opzionale di cluster larghi con PAM (k-medoids) in K parti fisse."
    )
    ap.add_argument("--xyz-dir", required=True, help="Cartella con .xyz (commento con E= HB= A= B= C=).")
    ap.add_argument("--out-dir", required=True, help="Cartella output.")
    # Filtro energia
    ap.add_argument("--dE-max", type=float, default=None, help="Usa solo (E - Emin) ≤ dE_max prima del clustering.")
    # Pesi distanza
    ap.add_argument("--w-rmsd", type=float, default=1.0, help="Peso RMSD normalizzata.")
    ap.add_argument("--w-abc",  type=float, default=1.0, help="Peso L1 su ABC normalizzate (MinMax).")
    # DBSCAN params
    ap.add_argument("--dbscan-eps", type=float, default=None, help="Se dato, usa eps fisso (altrimenti auto).")
    ap.add_argument("--dbscan-min-samples", type=int, default=None, help="Se dato, usa min_samples fisso (altrimenti auto).")
    ap.add_argument("--dbscan-auto", action="store_true", help="Forza ricerca automatica (ignora eps/min_samples).")
    ap.add_argument("--dbscan-noise-max", type=float, default=0.6, help="Massima frazione noise tollerata nello score.")
    ap.add_argument("--extent-penalty-weight", type=float, default=0.5, help="Penalità per cluster troppo estesi nello score.")
    # Split semplice con PAM
    ap.add_argument("--pam-split", action="store_true",
                    help="Se attivo, i cluster con estensione relativa > --max-cluster-extent-frac vengono partizionati con PAM in --pam-k sotto-cluster.")
    ap.add_argument("--pam-k", type=int, default=2, help="Numero di sotto-cluster per PAM (>=2).")
    ap.add_argument("--max-cluster-extent-frac", type=float, default=0.5,
                    help="Soglia (0..1) di larghezza relativa oltre cui scattare PAM split.")
    # Rappresentanti
    ap.add_argument("--rep-mode", choices=["minE","medoid","mds","core"], default="minE",
                    help="Criterio principale per rappresentanti (es. 'mds' seleziona il punto più vicino al centroide in MDS 2D).")
    ap.add_argument("--rep-include", choices=["mode","all3","list"], default="mode",
                    help="Quali rappresentanti includere come ★ e nei CSV flag: solo --rep-mode, tutti e tre (medoid,minE,minHB), oppure una lista personalizzata.")
    ap.add_argument("--rep-which", default="medoid,minE,minHB",
                    help="Usato se --rep-include list. Comma-separated tra: medoid,minE,minHB,mds,core")
    # Output/plot
    ap.add_argument("--csv-name", default="dbscan_results.csv")
    ap.add_argument("--clusters2d-png", default="clusters_2d.png")
    ap.add_argument("--energy2d-png",   default="energy_2d.png")
    ap.add_argument("--hb2d-png",       default="hb_2d.png")
    ap.add_argument("--clusters3d-png", default="clusters_3d.png")
    ap.add_argument("--energy3d-png",   default="energy_3d.png")
    ap.add_argument("--hb3d-png",       default="hb_3d.png")
    ap.add_argument("--copy-candidates", action="store_true",
                    help="Copia i rappresentanti in out_dir/candidates/.")
    ap.add_argument("--copy-candidates-which", choices=["all","medoid-only","mode-only"], default="all",
                    help="Quali rappresentanti copiare: 'all' (default: reps selezionati), 'medoid-only' (solo il medoid per cluster), 'mode-only' (solo quelli del --rep-mode).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cand_dir = os.path.join(args.out_dir, "candidates")
    if args.copy_candidates:
        os.makedirs(cand_dir, exist_ok=True)

    # Caricamento
    files = sorted([f for f in os.listdir(args.xyz_dir) if f.lower().endswith(".xyz")])
    if not files:
        raise SystemExit("Nessun .xyz trovato.")

    rows = []; coords_list = []; ABC_list = []
    for fn in files:
        path = os.path.join(args.xyz_dir, fn)
        try:
            coords, atom_lines, meta = parse_xyz(path)
        except Exception as e:
            print(f"[WARN] salto {fn}: {e}"); continue
        if meta["E"] is None or meta["A"] is None or meta["B"] is None or meta["C"] is None:
            print(f"[WARN] salto {fn}: mancano E/A/B/C nel commento"); continue
        rows.append({"file": fn, "energy": meta["E"], "hb": (np.nan if meta["HB"] is None else meta["HB"]),
                     "A": meta["A"], "B": meta["B"], "C": meta["C"], "nat": coords.shape[0]})
        coords_list.append(coords)
        ABC_list.append([meta["A"], meta["B"], meta["C"]])

    if not rows:
        raise SystemExit("Nessuna struttura valida dopo parsing.")

    df = pd.DataFrame(rows).reset_index(drop=True)

    # Omogeneità numero di atomi per RMSD
    if len(set(df["nat"])) != 1:
        raise SystemExit("Gli .xyz non hanno lo stesso numero di atomi. Uniforma (oppure chiedi una variante con assignment).")

    # Filtro energetico
    if args.dE_max is not None:
        Emin = float(df["energy"].min())
        mask = (df["energy"] - Emin) <= float(args.dE_max)
        df = df.loc[mask].reset_index(drop=True)
        coords_list = [coords_list[i] for i in range(len(mask)) if mask.iloc[i]]
        ABC_list    = [ABC_list[i]    for i in range(len(mask)) if mask.iloc[i]]

    ABC_arr = np.asarray(ABC_list, dtype=float)

    # Matrice D
    D, abc_scaler, max_rmsd = distance_matrix_rmsd_abc(
        coords_list, ABC_arr, w_rmsd=float(args.w_rmsd), w_abc=float(args.w_abc)
    )

    # DBSCAN
    if args.dbscan_auto or args.dbscan_eps is None or args.dbscan_min_samples is None:
        eps, ms, auto = dbscan_auto_params_precomputed(
            D, min_samples_grid=(3,4,5,6,8,10),
            noise_max=args.dbscan_noise_max,
            extent_penalty_weight=args.extent_penalty_weight
        )
        model = DBSCAN(eps=float(eps), min_samples=int(ms), metric="precomputed").fit(D)
        labels = model.labels_
        details = {"method":"dbscan-auto", "eps": float(eps), "min_samples": int(ms),
                   "silhouette": silhouette_precomputed_nonnoise(D, labels),
                   "noise_frac": float(np.mean(labels == -1)),
                   "n_clusters": len(set(labels) - {-1}),
                   "extent": cluster_relative_extent(D, labels)}
    else:
        eps = float(args.dbscan_eps); ms = int(args.dbscan_min_samples)
        model = DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit(D)
        labels = model.labels_
        details = {"method":"dbscan-fixed", "eps": eps, "min_samples": ms,
                   "silhouette": silhouette_precomputed_nonnoise(D, labels),
                   "noise_frac": float(np.mean(labels == -1)),
                   "n_clusters": len(set(labels) - {-1}),
                   "extent": cluster_relative_extent(D, labels)}

    # ========= SPLIT SEMPLICE CON PAM (K-MEDOIDS) =========
    if getattr(args, "pam_split", False):
        labs = labels.copy()
        next_label = (max([l for l in labs if l != -1], default=-1) + 1)
        G = global_diameter(D) + 1e-12
        for lab in sorted(set(labs) - {-1}):
            idxs = np.where(labs == lab)[0]
            if idxs.size < max(2, args.pam_k):
                continue
            sub = D[np.ix_(idxs, idxs)]
            local = float(np.max(sub))
            rel_extent = local / G
            if rel_extent > float(args.max_cluster_extent_frac) and args.pam_k >= 2:
                labels_local, _ = pam_kmedoids(sub, int(args.pam_k))
                unique_local = sorted(set(labels_local.tolist()))
                local_to_global = {}
                for u in unique_local:
                    local_to_global[u] = next_label
                    next_label += 1
                for pos, ii in enumerate(idxs):
                    labs[ii] = local_to_global[int(labels_local[pos])]
        labels = labs
        sil = silhouette_precomputed_nonnoise(D, labels)
        details.update({"post_split_silhouette": sil,
                        "post_split_noise_frac": float(np.mean(labels == -1)),
                        "post_split_n_clusters": len(set(labels) - {-1}),
                        "post_split_extent": cluster_relative_extent(D, labels),
                        "split_mode": f"PAM(k={int(args.pam_k)})"})

    # Embedding MDS 2D e 3D (necessario PRIMA di usare rep_mode=mds)
    X2 = mds_embed_2d(D)
    X3 = mds_embed_3d(D)

    # ======= Rappresentanti (multipli) =======
    # principale (rispetta --rep-mode, incluso 'mds')
    if args.rep_mode == "minE":
        reps_primary = pick_cluster_candidates_minE(df["energy"].to_numpy(float), labels)
    elif args.rep_mode == "medoid":
        reps_primary = pick_cluster_medoids(D, labels)
    elif args.rep_mode == "mds":
        reps_primary = pick_cluster_mds_center(X2, labels)
    elif args.rep_mode == "core":
        core_idx = getattr(model, "core_sample_indices_", np.array([], dtype=int))
        reps_primary = pick_cluster_densest_core(D, labels, core_idx, float(details.get("eps", eps)))
    else:
        reps_primary = pick_cluster_candidates_minE(df["energy"].to_numpy(float), labels)

    # altri set disponibili per flag/copie
    reps_med   = pick_cluster_medoids(D, labels)
    reps_minE  = pick_cluster_candidates_minE(df["energy"].to_numpy(float), labels)
    reps_minHB = pick_cluster_minHB(df["hb"].to_numpy(float), labels)
    reps_mds   = pick_cluster_mds_center(X2, labels)
    core_idx   = getattr(model, "core_sample_indices_", np.array([], dtype=int))
    reps_core  = pick_cluster_densest_core(D, labels, core_idx, float(details.get("eps", eps)))

    # comporre reps secondo --rep-include
    reps_to_unify: List[Dict[int,int]] = []
    if args.rep_include == "mode":
        reps_to_unify.append(reps_primary)
    elif args.rep_include == "all3":
        reps_to_unify.extend([reps_med, reps_minE, reps_minHB])
    else:  # list
        raw_selected = [s.strip() for s in args.rep_which.split(",") if s.strip()]
        for s in raw_selected:
            sl = s.lower()
            if sl == "medoid":
                reps_to_unify.append(reps_med)
            elif sl in ("mine","min_e","minenergy","min energy") or s == "minE":
                reps_to_unify.append(reps_minE)
            elif sl == "minhb":
                reps_to_unify.append(reps_minHB)
            elif sl == "mds":
                reps_to_unify.append(reps_mds)
            elif sl == "core":
                reps_to_unify.append(reps_core)

    reps_multi: Dict[int, List[int]] = unify_representatives(len(df), reps_to_unify if reps_to_unify else [reps_primary])

    # === Plot 2D ===
    plot_clusters_on_mds_2d(X2, labels, os.path.join(args.out_dir, args.clusters2d_png),
                            title=f"DBSCAN clusters (MDS 2D) — reps: {args.rep_include}",
                            representatives=reps_multi)
    plot_2d_colored(X2, df["energy"].to_numpy(float),
                    os.path.join(args.out_dir, args.energy2d_png),
                    label="Energy (visual)", title="MDS 2D — Energy")
    plot_2d_colored(X2, df["hb"].to_numpy(float),
                    os.path.join(args.out_dir, args.hb2d_png),
                    label="HB", title="MDS 2D — HB")

    # === Plot 3D ===
    plot_clusters_on_mds_3d(X3, labels, os.path.join(args.out_dir, args.clusters3d_png),
                            title=f"DBSCAN clusters (MDS 3D) — reps: {args.rep_include}",
                            representatives=reps_multi)
    plot_3d_colored(X3, df["energy"].to_numpy(float),
                    os.path.join(args.out_dir, args.energy3d_png),
                    label="Energy (visual)", title="MDS 3D — Energy")
    plot_3d_colored(X3, df["hb"].to_numpy(float),
                    os.path.join(args.out_dir, args.hb3d_png),
                    label="HB", title="MDS 3D — HB")

    # ======= CSV =======
    out_csv = os.path.join(args.out_dir, args.csv_name)
    df_out = df.copy()
    df_out["plot2d_x"] = X2[:,0]; df_out["plot2d_y"] = X2[:,1]
    df_out["plot3d_x"] = X3[:,0]; df_out["plot3d_y"] = X3[:,1]; df_out["plot3d_z"] = X3[:,2]
    df_out["cluster"] = labels

    # Flag per tipo di rappresentante
    is_rep_any   = np.zeros(len(df_out), dtype=bool)
    is_rep_med   = np.zeros(len(df_out), dtype=bool)
    is_rep_minE  = np.zeros(len(df_out), dtype=bool)
    is_rep_minHB = np.zeros(len(df_out), dtype=bool)
    is_rep_mds   = np.zeros(len(df_out), dtype=bool)
    is_rep_core  = np.zeros(len(df_out), dtype=bool)

    for lab, idx in reps_med.items():
        if 0 <= idx < len(df_out): is_rep_med[idx] = True
    for lab, idx in reps_minE.items():
        if 0 <= idx < len(df_out): is_rep_minE[idx] = True
    for lab, idx in reps_minHB.items():
        if 0 <= idx < len(df_out): is_rep_minHB[idx] = True
    for lab, idx in reps_mds.items():
        if 0 <= idx < len(df_out): is_rep_mds[idx] = True
    for lab, idx in reps_core.items():
        if 0 <= idx < len(df_out): is_rep_core[idx] = True

    is_rep_any = is_rep_med | is_rep_minE | is_rep_minHB | is_rep_mds | is_rep_core

    df_out["is_rep_any"]    = is_rep_any
    df_out["is_rep_medoid"] = is_rep_med
    df_out["is_rep_minE"]   = is_rep_minE
    df_out["is_rep_minHB"]  = is_rep_minHB
    df_out["is_rep_mds"]    = is_rep_mds
    df_out["is_rep_core"]   = is_rep_core

    # Parametri
    for k, v in details.items():
        df_out[f"param_{k}"] = v if not isinstance(v, (np.floating, np.integer)) else float(v)
    df_out["param_w_rmsd"] = float(args.w_rmsd)
    df_out["param_w_abc"]  = float(args.w_abc)
    df_out["param_max_rmsd_for_norm"] = float(max_rmsd)
    df_out["rep_mode"] = args.rep_mode
    df_out["rep_include"] = args.rep_include
    df_out["rep_which"] = (args.rep_which if args.rep_include == "list" else "")
    df_out["pam_split_enabled"] = bool(getattr(args, "pam_split", False))
    df_out["pam_k"] = int(getattr(args, "pam_k", 0))
    df_out["pam_max_extent_frac"] = float(getattr(args, "max_cluster_extent_frac", 0.0))

    df_out.to_csv(out_csv, index=False, float_format="%.8f")
    print(f"[OK] CSV salvato: {out_csv}")

    # ======= Copia dei rappresentanti =======
    if args.copy_candidates:
        copied = 0

        # 1) Quali indici copiare?
        if args.copy_candidates_which == "medoid-only":
            indices_to_copy = sorted(set(pick_cluster_medoids(D, labels).values()))
        elif args.copy_candidates_which == "mode-only":
            indices_to_copy = sorted(set(pick_cluster_mds_center(X2, labels).values()
                                         if args.rep_mode == "mds" else
                                         pick_cluster_candidates_minE(df["energy"].to_numpy(float), labels).values()
                                         if args.rep_mode == "minE" else
                                         pick_cluster_medoids(D, labels).values()
                                         if args.rep_mode == "medoid" else
                                         pick_cluster_densest_core(D, labels,
                                                                   getattr(model, "core_sample_indices_", np.array([], dtype=int)),
                                                                   float(details.get("eps", eps))).values()))
        else:  # "all"
            # usa reps_multi già de-duplicato
            indices_to_copy = sorted({i for _, lst in reps_multi.items() for i in lst})

        # 2) Copia con suffissi corretti
        for idx in indices_to_copy:
            if not (0 <= idx < len(df_out)):
                continue
            fname = df_out.loc[idx, "file"]
            src = os.path.join(args.xyz_dir, fname)
            lab = int(df_out.loc[idx, "cluster"])
            lab_str = "noise" if lab == -1 else f"cluster_{lab}"

            if args.copy_candidates_which == "medoid-only":
                suffixes = ["medoid"]  # forza solo 'medoid'
            elif args.copy_candidates_which == "mode-only":
                # suffisso coerente con la modalità
                mode = args.rep_mode
                suffixes = [mode]
            else:
                # mantieni i tag informativi se 'all'
                suffixes = []
                if df_out.loc[idx, "is_rep_medoid"]: suffixes.append("medoid")
                if df_out.loc[idx, "is_rep_minE"]:   suffixes.append("minE")
                if df_out.loc[idx, "is_rep_minHB"]:  suffixes.append("minHB")
                if df_out.loc[idx, "is_rep_mds"]:    suffixes.append("mds")
                if df_out.loc[idx, "is_rep_core"]:   suffixes.append("core")
                if not suffixes:
                    suffixes = ["rep"]

            for sfx in suffixes:
                dst = os.path.join(cand_dir, f"{lab_str}__{sfx}__{fname}")
                try:
                    shutil.copyfile(src, dst); copied += 1
                except Exception as e:
                    print(f"[WARN] non copio {fname} ({sfx}): {e}")

        print(f"[OK] Copiati {copied} file in: {cand_dir}")

    print("[DONE] DBSCAN clustering completato.")

if __name__ == "__main__":
    main()
