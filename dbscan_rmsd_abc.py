#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import shutil
from typing import List, Tuple, Dict, Optional

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
    D = np.diag([1.0, 1.0, np.sign(d)])
    U = V @ D @ Wt
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
# MDS: 2D e 3D
# =========================
def mds_embed_2d(D: np.ndarray) -> np.ndarray:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto")
    return mds.fit_transform(D)

def mds_embed_3d(D: np.ndarray) -> np.ndarray:
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=0, normalized_stress="auto")
    return mds.fit_transform(D)

# =========================
# PLOT 2D: clusters + colored
# =========================
def plot_clusters_on_mds_2d(X2: np.ndarray, labels: np.ndarray, out_png: str, title: str,
                            representatives: Optional[Dict[int,int]] = None):
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
    if representatives:
        star_labeled = False
        for lab, idx in representatives.items():
            if not (0 <= idx < X2.shape[0]): continue
            x, y = X2[idx, 0], X2[idx, 1]
            col = color_map.get(int(lab), "#000000")
            lbl = "representative (★)" if not star_labeled else None
            ax.scatter([x], [y], s=180, marker="*", c=[col],
                       edgecolors="black", linewidths=0.8, zorder=5, label=lbl)
            star_labeled = True
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2"); ax.set_title(title)
    # legenda fuori a destra
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

# =========================
# PLOT 3D: clusters + colored
# =========================
def plot_clusters_on_mds_3d(X3: np.ndarray, labels: np.ndarray, out_png: str, title: str,
                            representatives: Optional[Dict[int,int]] = None):
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
    if representatives:
        star_labeled = False
        for lab, idx in representatives.items():
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
    # legenda fuori
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
# Split ricorsivo cluster “larghi”
# =========================
def recursive_split_large_clusters(D: np.ndarray,
                                   labels: np.ndarray,
                                   max_extent_frac: float,
                                   min_size: int,
                                   max_levels: int,
                                   min_samples_grid=(3,4,5,6,8,10),
                                   noise_max: float = 0.6,
                                   extent_penalty_weight: float = 0.5) -> np.ndarray:
    labs = np.asarray(labels).copy()
    next_label = (max([l for l in labs if l != -1], default=-1) + 1)
    def _split(indices: np.ndarray, level: int):
        nonlocal labs, next_label
        if level >= max_levels or indices.size < min_size:
            return
        sub = D[np.ix_(indices, indices)]
        extent = float(np.max(sub)) / (global_diameter(D) + 1e-12)
        if extent <= max_extent_frac:
            return
        eps, ms, _ = dbscan_auto_params_precomputed(sub, min_samples_grid=min_samples_grid,
                                                    noise_max=noise_max,
                                                    extent_penalty_weight=extent_penalty_weight)
        sub_model = DBSCAN(eps=float(eps), min_samples=int(ms), metric="precomputed").fit(sub)
        sub_labels = sub_model.labels_
        uniq = sorted(set(sub_labels) - {-1})
        if len(uniq) <= 1:
            return
        for u in uniq:
            mask_u = (sub_labels == u)
            labs[indices[mask_u]] = next_label
            _split(indices[mask_u], level+1)
            next_label += 1
        labs[indices[sub_labels == -1]] = -1
    for lab in sorted(set(labs) - {-1}):
        idxs = np.where(labs == lab)[0]
        _split(idxs, level=0)
    return labs

# =========================
# Representative pickers
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
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser(
        description="DBSCAN su distanza RMSD_norm + L1(ABC_norm) con auto-ricerca (eps,min_samples), split ricorsivo, reps selezionabili e plot 2D/3D."
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
    # Split ricorsivo
    ap.add_argument("--split-large-clusters", action="store_true", help="Abilita split ricorsivo di cluster larghi (diametro relativo).")
    ap.add_argument("--max-cluster-extent-frac", type=float, default=0.5, help="Soglia di larghezza relativa per lo split.")
    ap.add_argument("--sub-min-size", type=int, default=12, help="Dimensione minima cluster per suddividerlo.")
    ap.add_argument("--max-subdiv-levels", type=int, default=1, help="Profondità massima di suddivisione.")
    # Rappresentante
    ap.add_argument("--rep-mode", choices=["minE","medoid","mds","core"], default="minE",
                    help="Criterio di rappresentante per cluster.")
    # Output/plot
    ap.add_argument("--csv-name", default="dbscan_results.csv")
    ap.add_argument("--clusters2d-png", default="clusters_2d.png")
    ap.add_argument("--energy2d-png",   default="energy_2d.png")
    ap.add_argument("--hb2d-png",       default="hb_2d.png")
    ap.add_argument("--clusters3d-png", default="clusters_3d.png")
    ap.add_argument("--energy3d-png",   default="energy_3d.png")
    ap.add_argument("--hb3d-png",       default="hb_3d.png")
    ap.add_argument("--copy-candidates", action="store_true", help="Copia i rappresentanti in out_dir/candidates/.")
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

    # Split ricorsivo (opzionale)
    if args.split_large_clusters:
        labels = recursive_split_large_clusters(
            D, labels,
            max_extent_frac=float(args.max_cluster_extent_frac),
            min_size=int(args.sub_min_size),
            max_levels=int(args.max_subdiv_levels),
            min_samples_grid=(3,4,5,6,8,10),
            noise_max=args.dbscan_noise_max,
            extent_penalty_weight=args.extent_penalty_weight
        )
        sil = silhouette_precomputed_nonnoise(D, labels)
        details.update({"post_split_silhouette": sil,
                        "post_split_noise_frac": float(np.mean(labels == -1)),
                        "post_split_n_clusters": len(set(labels) - {-1}),
                        "post_split_extent": cluster_relative_extent(D, labels)})

    # Embedding MDS 2D e 3D
    X2 = mds_embed_2d(D)
    X3 = mds_embed_3d(D)

    # Rappresentanti
    rep_mode = args.rep_mode
    if rep_mode == "minE":
        reps_local = pick_cluster_candidates_minE(df["energy"].to_numpy(float), labels)
    elif rep_mode == "medoid":
        reps_local = pick_cluster_medoids(D, labels)
    elif rep_mode == "mds":
        reps_local = pick_cluster_mds_center(X2, labels)  # centroide nello spazio 2D (come richiesto)
    elif rep_mode == "core":
        core_idx = getattr(model, "core_sample_indices_", np.array([], dtype=int))
        reps_local = pick_cluster_densest_core(D, labels, core_idx, float(details.get("eps", eps)))
    else:
        reps_local = pick_cluster_candidates_minE(df["energy"].to_numpy(float), labels)

    # === Plot 2D ===
    plot_clusters_on_mds_2d(X2, labels, os.path.join(args.out_dir, args.clusters2d_png),
                            title=f"DBSCAN clusters (MDS 2D) — reps: {rep_mode}",
                            representatives=reps_local)
    plot_2d_colored(X2, df["energy"].to_numpy(float),
                    os.path.join(args.out_dir, args.energy2d_png),
                    label="Energy (visual)", title="MDS 2D — Energy")
    plot_2d_colored(X2, df["hb"].to_numpy(float),
                    os.path.join(args.out_dir, args.hb2d_png),
                    label="HB", title="MDS 2D — HB")

    # === Plot 3D ===
    plot_clusters_on_mds_3d(X3, labels, os.path.join(args.out_dir, args.clusters3d_png),
                            title=f"DBSCAN clusters (MDS 3D) — reps: {rep_mode}",
                            representatives=reps_local)
    plot_3d_colored(X3, df["energy"].to_numpy(float),
                    os.path.join(args.out_dir, args.energy3d_png),
                    label="Energy (visual)", title="MDS 3D — Energy")
    plot_3d_colored(X3, df["hb"].to_numpy(float),
                    os.path.join(args.out_dir, args.hb3d_png),
                    label="HB", title="MDS 3D — HB")

    # CSV
    out_csv = os.path.join(args.out_dir, args.csv_name)
    df_out = df.copy()
    df_out["plot2d_x"] = X2[:,0]; df_out["plot2d_y"] = X2[:,1]
    df_out["plot3d_x"] = X3[:,0]; df_out["plot3d_y"] = X3[:,1]; df_out["plot3d_z"] = X3[:,2]
    df_out["cluster"] = labels
    is_rep = np.zeros(len(df_out), dtype=bool)
    for lab, loc_idx in reps_local.items():
        if 0 <= loc_idx < len(df_out): is_rep[loc_idx] = True
    df_out["is_representative"] = is_rep
    df_out["rep_mode"] = rep_mode
    for k, v in details.items():
        df_out[f"param_{k}"] = v if not isinstance(v, (np.floating, np.integer)) else float(v)
    df_out["param_w_rmsd"] = float(args.w_rmsd)
    df_out["param_w_abc"]  = float(args.w_abc)
    df_out["param_max_rmsd_for_norm"] = float(max_rmsd)
    df_out.to_csv(out_csv, index=False, float_format="%.8f")
    print(f"[OK] CSV salvato: {out_csv}")

    # Copia rappresentanti
    if args.copy_candidates:
        copied = 0
        for lab, loc_idx in reps_local.items():
            fname = df_out.loc[loc_idx, "file"]
            src = os.path.join(args.xyz_dir, fname)
            lab_str = "noise" if lab == -1 else f"cluster_{lab}"
            dst = os.path.join(cand_dir, f"{lab_str}__{fname}")
            try:
                shutil.copyfile(src, dst); copied += 1
            except Exception as e:
                print(f"[WARN] non copio {fname}: {e}")
        print(f"[OK] Copiati {copied}/{len(reps_local)} rappresentanti in: {cand_dir}")

    print("[DONE] DBSCAN clustering completato.")

if __name__ == "__main__":
    main()

