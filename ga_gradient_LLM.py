#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import re
import math
import random
import subprocess
import argparse
import uuid
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from typing import Optional, List

# === AGGIUNTE PER OPERATORE INDUZIONE (LLM LOCALE) ===
import json
import time
import requests
# =====================================================

# --- Parametri di configurazione (come originali) ---
INPUT_FILE = ""          # .gjf di riferimento
TMP_DIR = "tmp"
GENERATIONS_DIR = ""

NUM_GENERAZIONI = 20
POPOLAZIONE_INIZIALE = 20
POPOLAZIONE_TARGET = 10

# Oscillazioni (come prima)
BASE_RATE_MUTATION = 0.45
BASE_RATE_CROSSOVER = 0.8
DELTA_RATE_MUTATION = 0.1
DELTA_RATE_CROSSOVER = 0.1
NUM_OSCILLATIONS = 2

# --- H-bond objective ---
HB_SPHERE = 2.5
HB_BONUS_PER_BOND = 0.02
HB_MUTUAL_PENALTY = 0.05

# --- H-bond biforcati ---
HB_CONTACT_THRESHOLD = -0.3   # soglia energia Ek per considerare un H-bond "attivo"

# --- Pesi diversi per tipologia di H-bond ---
HB_EPS_BASE = 1.0
HB_EPS_OHN  = 1.1
HB_EPS_OHO  = 1.2
HB_EPS_NHO  = 1.1
HB_EPS_SHO  = 1.0
HB_EPS_SHN  = 1.0
HB_EPS_OHS  = 1.2
HB_EPS_NHS  = 1.2

# --- H–H penalty params ---
HH_EPS   = 0.1
HH_ALPHA = 15.0
HH_XM    = 1.5  # Å

# --- SBX ---
SBX_ETA = 15.0
ANGLE_LOW = 0.0
ANGLE_HIGH = 360.0

# --- Orientazioni parse ---
USE_STANDARD_ORIENTATION = True

# (legacy, non usati)
PAIR_SPHERE = 2.5
NEAR_PAIRS = []
NEAR_WEIGHTS = []

# --- “GENI” letti dal gjf ---
GENI = []

# --- Mappa simbolo -> Z ---
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}

# --- Raggi covalenti (Å) ---
COV_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.01}

# --- Raggi di van der Waals (Å) (Bondi) ---
VDW_RADII = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 16: 1.80}

# =========================
# OBIETTIVI MULTI-OUTPUT
# =========================
# Ordine di minimizzazione richiesto:
# 1) HBond fitness (fitness_hbond)
# 2) Energia (fitness_energy)
# 3) GRMS (fitness_grms)
# 4) GMAX (fitness_gmax)
OBJECTIVES: List[str] = ["fitness_hbond", "fitness_hb_bifork", "fitness_energy", "fitness_grms", "fitness_gmax"]

# === COSTANTI OPERATORE INDUZIONE ===========================================
P_INDUZIONE_DEFAULT = 0.15              # probabilità di usare l'operatore per ogni figlio
INDUZIONE_MODEL_DEFAULT = "mistral:7b-instruct"
INDUZIONE_TIMEOUT_S = 20.0              # timeout richiesta al modello locale
INDUZIONE_HISTORY_K = 20                # quanti individui passare nello storico
INDUZIONE_STEP_DEG = 20.0               # manteniamo coerenza con generate_random_allele_discrete
INDUZIONE_MAX_REASK = 1                 # quante volte riprovare se l'output non è valido
# ============================================================================

# ========== UTIL =============================================================

def parse_coords_from_gjf(gjf_path):
    rows = []
    float_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
    token_re = re.compile(rf"^\s*([A-Za-z]+|\d+)\s+({float_re})\s+({float_re})\s+({float_re})\s*$")
    with open(gjf_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = token_re.match(line)
            if not m:
                continue
            t, xs, ys, zs = m.groups()
            if t.isdigit():
                Z = int(t)
            else:
                Z = ELEMENT_Z.get(t, None)
                if Z is None:
                    continue
            rows.append((Z, float(xs), float(ys), float(zs)))
    return rows

def strip_gene_lines(gjf_path):
    lines = []
    with open(gjf_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().upper().startswith("GENE"):
                continue
            lines.append(line)
    return "".join(lines)

def cov_bonded(Z1, Z2, d, tol=0.4):
    r1 = COV_RADII.get(Z1, 0.75)
    r2 = COV_RADII.get(Z2, 0.75)
    return d <= (r1 + r2 + tol)

def dist3(pa, pb):
    dx = pa[0]-pb[0]; dy = pa[1]-pb[1]; dz = pa[2]-pb[2]
    return (dx*dx + dy*dy + dz*dz)**0.5

def build_bond_graph(coords):
    n = len(coords)
    P = [(x,y,z) for (_,x,y,z) in coords]
    Z = [Z for (Z,_,_,_) in coords]
    bonds = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dij = dist3(P[i], P[j])
            if cov_bonded(Z[i], Z[j], dij):
                bonds[i].add(j)
                bonds[j].add(i)
    return bonds

def same_topology(bonds_a, bonds_b):
    if len(bonds_a) != len(bonds_b):
        return False
    for s1, s2 in zip(bonds_a, bonds_b):
        if s1 != s2:
            return False
    return True

def tweak_some_alleles_random(alleles, geni, k: Optional[int]=None, eps_deg=1e-6):
    n = len(alleles)
    if n == 0:
        return alleles, []
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

def snapshot_for_rescue(ind):
    return {
        "alleli": ind["alleli"][:],
        "fitness_hbond": ind["fitness_hbond"],
        # in snapshot_for_rescue
        "fitness_hb_bifork": ind.get("fitness_hb_bifork", float("inf")),
        "fitness_energy": ind["fitness_energy"],
        "fitness_grms": ind["fitness_grms"],
        "fitness_gmax": ind["fitness_gmax"],
        "num_atoms": ind.get("num_atoms"),
        "xyz_lines": ind.get("xyz_lines", [])[:],
        "helped": ind.get("helped", False),
        "hb_details": copy.deepcopy(ind.get("hb_details", {})),
    }

def clone_from_rescue(ind, rescue_pool):
    if not rescue_pool:
        return False
    donor = random.choice(rescue_pool)
    ind["alleli"]         = donor["alleli"][:]
    for k in OBJECTIVES:
        ind[k] = donor[k]
    ind["num_atoms"]      = donor.get("num_atoms")
    ind["xyz_lines"]      = donor.get("xyz_lines", [])[:]
    ind["xyz_file"]       = None
    ind["helped"]         = donor.get("helped", False)
    ind["hb_details"]     = copy.deepcopy(donor.get("hb_details", {}))
    return True

def parse_last_orientation_coords(log_file, use_standard=True):
    def _is_dash(s: str) -> bool:
        s = s.strip()
        return bool(s) and set(s) == {'-'}
    with open(log_file, 'r', errors='ignore') as f:
        text = f.read()
    key = "Standard orientation:" if use_standard else "Principal axis orientation:"
    idx = text.rfind(key)
    if idx == -1:
        return []
    lines = text[idx:].splitlines()
    dash_idxs = [i for i, l in enumerate(lines) if _is_dash(l)]
    if len(dash_idxs) < 2:
        return []
    start = dash_idxs[1] + 1
    rows = []
    for line in lines[start:]:
        if _is_dash(line):
            break
        toks = line.split()
        try:
            if use_standard:
                if len(toks) >= 6 and toks[1].isdigit():
                    Z = int(toks[1])
                    x = float(toks[3]); y = float(toks[4]); z = float(toks[5])
                    rows.append((Z, x, y, z))
            else:
                if len(toks) >= 5 and toks[1].isdigit():
                    Z = int(toks[1])
                    x = float(toks[2]); y = float(toks[3]); z = float(toks[4])
                    rows.append((Z, x, y, z))
        except ValueError:
            continue
    if not rows:
        if use_standard:
            return parse_last_orientation_coords(log_file, use_standard=False)
        else:
            return []
    return rows

# --- ENERGY PARSER (SCF Done / HF= robusto) ---
def parse_fitness(log_file):
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()
    scf_matches = re.findall(
        r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
        content
    )
    if scf_matches:
        val = scf_matches[-1].replace('D', 'E').replace('d', 'E')
        try: return float(val)
        except: pass
    compact = re.sub(r"\s+", "", content)
    hf_matches = re.findall(r"HF=([+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)\\", compact)
    if hf_matches:
        raw = hf_matches[-1].replace('D', 'E').replace('d', 'E')
        try: return float(raw)
        except: pass
    m = re.search(r"HF=\s*([+-]?\s*\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)", content, flags=re.IGNORECASE|re.MULTILINE)
    if m:
        raw = m.group(1).replace(" ", "").replace('D','E').replace('d','E')
        try: return float(raw)
        except: pass
    other_matches = re.findall(r"\bEnergy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)", content, flags=re.IGNORECASE)
    if other_matches:
        val = other_matches[-1].replace('D', 'E').replace('d', 'E')
        try:
            x = float(val)
            if abs(x) > 1e-12:
                return x
        except: pass
    print(f"[parse_fitness] Energia non trovata in {log_file}")
    return None

# --- CARTESIAN FORCES PARSER (GRMS, GMAX) ---
_CF_LINE = re.compile(
    r"Cartesian\s+Forces:\s*Max\s+([-\d\.Ee+]+)\s*RMS\s+([-\d\.Ee+]+)",
    flags=re.IGNORECASE
)

def parse_cartesian_forces_rms(log_file: str):
    try:
        with open(log_file, "r", errors="ignore") as f:
            txt = f.read()
    except:
        return None, None
    m = _CF_LINE.search(txt)
    if not m:
        compact = re.sub(r"\s+", " ", txt)
        m = _CF_LINE.search(compact)
        if not m:
            return None, None
    gmax = float(m.group(1))
    grms = float(m.group(2))
    return grms, gmax

def parse_xyz_from_log(log_file):
    # riutilizza il parser robusto già presente
    INV_Z = {v:k for k,v in ELEMENT_Z.items()}
    coords = parse_last_orientation_coords(log_file, use_standard=True)
    if not coords:
        coords = parse_last_orientation_coords(log_file, use_standard=False)
    if not coords:
        return None, []
    num_atoms = len(coords)
    xyz_lines = []
    for Z, x, y, z in coords:
        sym = INV_Z.get(Z, str(Z))
        xyz_lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    return num_atoms, xyz_lines
    
def parse_rotational_constants_mhz(log_file):
    A = B = C = None
    try:
        with open(log_file, 'r', errors='ignore') as f:
            for line in f:
                if "Rotational constants" in line and "MHZ" in line.upper():
                    parts = re.findall(r"([-\d\.EeDd\+]+)", line)
                    nums = [float(p.replace('D','E').replace('d','E')) for p in parts[-3:]] if len(parts) >= 3 else []
                    if len(nums) == 3:
                        A, B, C = nums
                        break
                    nxt = next(f, "")
                    parts = nxt.strip().split()
                    if len(parts) >= 3:
                        A = float(parts[0]); B = float(parts[1]); C = float(parts[2])
                    break
    except Exception:
        pass
    return A, B, C

def read_reference_file(filepath):
    with open(filepath, 'r') as f:
        return f.readlines()

def remove_frozen_substring(lines):
    new_lines = []
    for line in lines:
        _ = re.sub(r'(?i)frozen,?', '', line)
        new_lines.append(line)
    return new_lines

def cleanup_individual_tmp(tmp_dir, individual_id):
    base = f"individuo_{individual_id}"
    for ext in (".gjf", ".log", ".chk"):
        p = os.path.join(tmp_dir, base + ext)
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

def save_statistics(file_name, generations_data):
    fields = [
        "Generation",
        "Avg. HB", "MAX HB", "MIN HB",
        "Avg. HB Bif.", "MAX HB Bif.", "MIN HB Bif.",
        "Avg. Energy", "MAX Energy", "MIN Energy",
        "Avg. GRMS", "MAX GRMS", "MIN GRMS",
        "Avg. GMAX", "MAX GMAX", "MIN GMAX",
        "Mutation rate", "Crossover rate"
    ]
    with open(file_name, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in generations_data:
            writer.writerow(row)

def write_individual_file(directory, individual_id, content_lines):
    filename = os.path.join(directory, f"individuo_{individual_id}.gjf")
    with open(filename, 'w') as f:
        f.writelines(content_lines)
    return filename

def run_gdv(gjf_file, log_file):
    with open(gjf_file, 'r') as infile, open(log_file, 'w') as outfile:
        result = subprocess.run(["gdv"], stdin=infile, stdout=outfile, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        print(f"Errore nell'esecuzione di g16/gdv per {gjf_file}: {result.stderr}")
    return result.returncode

def circular_diff_deg(a, b):
    d = abs(a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d

def are_genotypes_similar(alleles1, alleles2, thr_deg):
    if len(alleles1) != len(alleles2):
        return False
    for aa, bb in zip(alleles1, alleles2):
        if circular_diff_deg(aa, bb) > thr_deg:
            return False
    return True

def parse_genes_from_gjf(gjf_path):
    genes = []
    pattern = re.compile(
        r'^\s*GENE[\w\-]*\s*\((?P<inside>[^)]*?)\)\s*=\s*(?P<rhs>.+?)\s*$',
        flags=re.IGNORECASE
    )
    per_re = re.compile(r'periodicity\s*=\s*(\d+)', flags=re.IGNORECASE)
    try:
        with open(gjf_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    inside = m.group("inside")
                    rhs = m.group("rhs").strip()
                    pm = per_re.search(inside or "")
                    if pm and rhs:
                        per = int(pm.group(1))
                        genes.append((per, rhs))
    except FileNotFoundError:
        pass
    return genes

def add_gene_lines(lines, geni, alleli):
    while lines and lines[-1].strip() == "":
        lines.pop()
    new_lines = list(lines)
    for idx, ((period, definition), allele) in enumerate(zip(geni, alleli), start=1):
        gene_line = f"GENE{idx}(Frozen,Value={allele:.4f}) = {definition}\n"
        new_lines.append(gene_line)
    new_lines.append("\n")
    return new_lines

def write_xyz_file(directory, individual_id, hb, energy, grms, gmax, rank, num_atoms, xyz_lines,
                   rotA=None, rotB=None, rotC=None, hb_bif=None):
    filename = os.path.join(directory, f"individuo_{individual_id}.xyz")
    with open(filename, 'w') as f:
        f.write(f"{num_atoms}\n")
        meta = f"HB={hb}  E={energy}  GRMS={grms}  GMAX={gmax}  Rank={rank}"
        if hb_bif is not None and math.isfinite(hb_bif):
            meta += f"  HBbif={hb_bif}"
        if rotA is not None and rotB is not None and rotC is not None:
            meta += f"  A={rotA}  B={rotB}  C={rotC}"
        f.write(meta + "\n")
        for line in xyz_lines:
            f.write(line + "\n")
    return filename

def cleanup_tmp(directory):
    for fname in os.listdir(directory):
        if fname.endswith((".gjf", ".log", ".chk")):
            os.remove(os.path.join(directory, fname))

# ========== H-BOND DETECTION ================================================

ACCEPTOR_ELEMENTS = (7, 8, 16)   # N, O, S

def identify_donors_acceptors(initial_coords, initial_bonds):
    Z = [Z for (Z,_,_,_) in initial_coords]
    donors = []
    donors_H = {}
    acceptors = [i for i,z in enumerate(Z) if z in (7,8,16)]
    for i, z in enumerate(Z):
        if z in (7,8,16):
            Hs = [j for j in initial_bonds[i] if Z[j] == 1]
            if Hs:
                donors.append(i)
                donors_H[i] = Hs

    print("Donatori (0-based):")
    for d in donors:
        print(f"  Atomo {d+1} (Z={Z[d]}) con H legati -> {[h+1 for h in donors_H[d]]}")
    print("Accettori (0-based):", [a+1 for a in acceptors])

    return donors, donors_H, acceptors

def graph_distance_leq(bonds, i, j, max_hops: int) -> bool:
    if bonds is None:
        return False
    if i == j:
        return True
    seen = {i}
    dq = deque([(i, 0)])
    while dq:
        u, d = dq.popleft()
        if d >= max_hops:
            continue
        for v in bonds[u]:
            if v == j:
                return True
            if v not in seen:
                seen.add(v)
                dq.append((v, d+1))
    return False

def evaluate_bifurcated_hbond_fitness(
    coords,
    donors, donors_H, acceptors,
    contact_threshold=HB_CONTACT_THRESHOLD,
    bonds=None,
    xm_scale: float = 1.0,   # <— fattore percentuale (1.10 = +10%)
    xm_add: float   = 0.1,   # <— offset in Å (0.10 = +0.10 Å)
    alpha_bif: float = 5.0  # <— curvatura (più basso = potenziale più “morbido”)
):
    """
    Conta quanti ACCETTORI sono 'biforcati', cioè hanno >=2 H-bond validi.
    Un H-bond è valido se Ek (dalla stessa hbond_pair_energy) <= contact_threshold.
    XM viene modificato come: XM' = XM*xm_scale + xm_add
    L'alpha usata è alpha_bif (indipendente dalla HB energetica standard).
    """
    if not coords:
        return 0.0, {"acceptor_counts": {}, "pairs": [], "threshold": contact_threshold,
                    "xm_scale": xm_scale, "xm_add": xm_add, "alpha_bif": alpha_bif}

    P = [(x,y,z) for (_,x,y,z) in coords]
    Z = [Z for (Z,_,_,_) in coords]
    n_atoms = len(P)

    donors_valid = [d for d in donors if 0 <= d < n_atoms]
    acc_valid    = [a for a in acceptors if 0 <= a < n_atoms]

    acc_counts = {a: 0 for a in acc_valid}
    pairs_info = []

    for D in donors_valid:
        Hs = [h for h in donors_H.get(D, []) if 0 <= h < n_atoms and Z[h] == 1]
        for H in Hs:
            pH = P[H]
            for A in acc_valid:
                if A == D:
                    continue

                # xm di base come nell'H-bond standard
                rD_vdw = VDW_RADII.get(Z[D], 0.75)
                rA_vdw = VDW_RADII.get(Z[A], 0.75)
                xm_base = 0.60 * (rD_vdw + rA_vdw)

                # modifica XM
                xm = xm_base * xm_scale + xm_add
                if xm <= 1e-12:
                    continue

                # stesse eps per tipologia D/A
                if Z[D] == 8 and Z[A] == 7:
                    eps_eff = HB_EPS_OHN
                elif Z[D] == 7 and Z[A] == 8:
                    eps_eff = HB_EPS_NHO
                elif Z[D] == 8 and Z[A] == 8:
                    eps_eff = HB_EPS_OHO
                elif Z[D] == 16 and Z[A] == 8:
                    eps_eff = HB_EPS_SHO
                elif Z[D] == 16 and Z[A] == 7:
                    eps_eff = HB_EPS_SHN
                elif Z[D] == 8 and Z[A] == 16:
                    eps_eff = HB_EPS_OHS
                elif Z[D] == 7 and Z[A] == 16:
                    eps_eff = HB_EPS_NHS
                else:
                    eps_eff = HB_EPS_BASE

                xk = dist3(pH, P[A])

                # usa alpha_bif (curvatura per la fitness biforcata)
                Ek = hbond_pair_energy(xk, xm, alpha=alpha_bif, eps=eps_eff)

                if Ek <= contact_threshold:
                    acc_counts[A] += 1
                    print(f"Trovato un candidato biforcato tra {D+1}-{H+1} - - - {A+1}")
                    pairs_info.append({
                        "D": D+1, "H": H+1, "A": A+1,
                        "Ek": Ek, "dHA": xk,
                        "xm_base": xm_base, "xm_used": xm,
                        "alpha_bif": alpha_bif
                    })

    n_bif = sum(1 for _, c in acc_counts.items() if c >= 2)
    print(f"-------------------------- N BIF. = {n_bif} --------------------------")
    fitness = -float(n_bif)

    details = {
        "acceptor_counts": {a+1: c for a, c in acc_counts.items()},
        "pairs": pairs_info,
        "n_bifurcated_acceptors": n_bif,
        "threshold": contact_threshold,
        "xm_scale": xm_scale,
        "xm_add": xm_add,
        "alpha_bif": alpha_bif
    }
    return fitness, details
    
def evaluate_hbond_fitness(coords, donors, donors_H, acceptors,
                           sphere=HB_SPHERE,
                           bonus=HB_BONUS_PER_BOND,
                           mutual_penalty=HB_MUTUAL_PENALTY,
                           bonds=None):
    if not coords:
        return 0.0, {"pairs": [], "mutual": [], "hh_penalty": {"sum": 0.0, "pairs": []}}, False

    P = [(x,y,z) for (_,x,y,z) in coords]
    Z = [Z for (Z,_,_,_) in coords]
    n_atoms = len(P)

    donors_valid = [d for d in donors if 0 <= d < n_atoms]
    acc_valid    = [a for a in acceptors if 0 <= a < n_atoms]

    total_E = 0.0
    pairs_info = []
    any_negative = False

    for D in donors_valid:
        rD_vdw = VDW_RADII.get(Z[D], 0.75)
        Hs = [h for h in donors_H.get(D, []) if 0 <= h < n_atoms and Z[h] == 1]
        for H in Hs:
            pH = P[H]
            for A in acc_valid:
                if A == D:
                    continue
                rA_vdw = VDW_RADII.get(Z[A], 0.75)
                xm = 0.60 * (rD_vdw + rA_vdw)
                xk = dist3(pH, P[A])

                if Z[D] == 8 and Z[A] == 7:
                    eps_eff = HB_EPS_OHN
                elif Z[D] == 7 and Z[A] == 8:
                    eps_eff = HB_EPS_NHO
                elif Z[D] == 8 and Z[A] == 8:
                    eps_eff = HB_EPS_OHO
                elif Z[D] == 16 and Z[A] == 8:
                    eps_eff = HB_EPS_SHO
                elif Z[D] == 16 and Z[A] == 7:
                    eps_eff = HB_EPS_SHN
                elif Z[D] == 8 and Z[A] == 16:
                    eps_eff = HB_EPS_OHS
                elif Z[D] == 7 and Z[A] == 16:
                    eps_eff = HB_EPS_NHS
                else:
                    eps_eff = HB_EPS_BASE

                Ek = hbond_pair_energy(xk, xm, alpha=15, eps=eps_eff)
                total_E += Ek
                if Ek < -0.3:
                    any_negative = True
                pairs_info.append({"D": D+1, "H": H+1, "A": A+1, "Z_D": Z[D], "Z_A": Z[A], "dHA": xk, "xm": xm, "Ek": Ek})

    all_H = [i for i,z in enumerate(Z) if z == 1]
    donor_H_set = set(h for hs in donors_H.values() for h in hs if 0 <= h < n_atoms)
    seen_pairs = set()
    hh_pairs_info = []
    hh_pen_sum = 0.0

    for hi in donor_H_set:
        for hj in all_H:
            if hj == hi:
                continue
            key = (hi, hj) if hi < hj else (hj, hi)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            if bonds is not None and graph_distance_leq(bonds, hi, hj, max_hops=4):
                continue
            xk = dist3(P[hi], P[hj])
            if xk < HH_XM:
                hh_pen_sum += 1e3
            hh_pairs_info.append({"H1": hi+1, "H2": hj+1, "dHH": xk, "xm": HH_XM, "alpha": HH_ALPHA, "eps": HH_EPS, "E_raw": 1e3 if xk<HH_XM else 0.0})

    total_E += hh_pen_sum

    details = {
        "pairs": pairs_info,
        "mutual": [],
        "hh_penalty": {"sum": hh_pen_sum, "pairs": hh_pairs_info}
    }
    return total_E, details, any_negative

def hbond_pair_energy(xk, xm, alpha=15.0, eps=1.0):
    if xm <= 1e-12:
        return 0.0
    t = xk / xm
    return eps * ( math.exp(alpha*(1.0 - t))
                   - (t*t - 2.0*t + 3.0) * math.exp(0.5*alpha*(1.0 - t)) )

# ========== SBX (angoli con wrap) ============================================

def _bounded_sbx(x1, x2, L, U, eta):
    if x1 > x2:
        x1, x2 = x2, x1
    if abs(x2 - x1) < 1e-12:
        return (x1 + x2) * 0.5
    u = random.random()
    beta = 1.0 + 2.0 * (x1 - L) / (x2 - x1)
    alpha = 2.0 - pow(beta, -(eta + 1.0))
    if u <= 1.0/alpha:
        betaq = pow(u * alpha, 1.0/(eta + 1.0))
    else:
        betaq = pow(1.0/(2.0 - u*alpha), 1.0/(eta + 1.0))
    child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
    beta = 1.0 + 2.0 * (U - x2) / (x2 - x1)
    alpha = 2.0 - pow(beta, -(eta + 1.0))
    if u <= 1.0/alpha:
        betaq = pow(u * alpha, 1.0/(eta + 1.0))
    else:
        betaq = pow(1.0/(2.0 - u*alpha), 1.0/(eta + 1.0))
    child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
    return child1 if random.random() < 0.5 else child2

def _wrap360(x):
    x = x % 360.0
    if x < 0: x += 360.0
    return x

_DISCRETE_PROBS_CACHE = {}

def generate_random_allele_discrete(periodicity: int, step_degrees: float = 20.0) -> float:
    w = 0.08
    b = 1/(2*math.pi) - w
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

def generate_random_allele(periodicity):
    return generate_random_allele_discrete(periodicity)

def resample_random_alleles(geni):
    return [generate_random_allele(period) for period, _ in geni]

def tweak_one_allele_random(alleles, geni, eps_deg=1e-6):
    if not alleles:
        return alleles, None
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

def sbx_crossover_angles(a1, a2, eta=SBX_ETA):
    delta = ((a2 - a1 + 540.0) % 360.0) - 180.0
    x1 = 0.0
    x2 = delta
    child_rel = _bounded_sbx(x1, x2, -180.0, 180.0, eta)
    child = a1 + child_rel
    return _wrap360(child)

def crossover_sbx(parent1, parent2, eta=SBX_ETA):
    alleles = []
    for a1, a2 in zip(parent1["alleli"], parent2["alleli"]):
        alleles.append(sbx_crossover_angles(a1, a2, eta=eta))
    return alleles

def mutate(alleles, mutation_rate, geni):
    new_alleles = []
    for allele, (period, _) in zip(alleles, geni):
        if random.random() < mutation_rate:
            new_alleles.append(generate_random_allele(period))
        else:
            new_alleles.append(allele)
    return new_alleles

# ========== NSGA-II (multi-obiettivo generico) ===============================

def dominates_generic(a, b, objectives: List[str]):
    # Minimizzazione per tutti gli obiettivi in 'objectives'
    a_vals = [a.get(k, float('inf')) for k in objectives]
    b_vals = [b.get(k, float('inf')) for k in objectives]
    if not all(math.isfinite(x) for x in a_vals + b_vals):
        return False
    not_worse = all(av <= bv for av, bv in zip(a_vals, b_vals))
    strictly_better = any(av < bv for av, bv in zip(a_vals, b_vals))
    return not_worse and strictly_better

def fast_non_dominated_sort(pop, objectives: List[str]):
    S = {i: [] for i in range(len(pop))}
    n_dom = [0]*len(pop)
    fronts = [[]]
    for i, p in enumerate(pop):
        S[i] = []
        n_dom[i] = 0
        for j, q in enumerate(pop):
            if i == j: continue
            if dominates_generic(p, q, objectives):
                S[i].append(j)
            elif dominates_generic(q, p, objectives):
                n_dom[i] += 1
        if n_dom[i] == 0:
            p["rank"] = 0
            fronts[0].append(i)
    f = 0
    while fronts[f]:
        Q = []
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

def crowding_distance(front, pop, objectives: List[str]):
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
        for k in range(1, len(front_sorted)-1):
            prev_val = pop[front_sorted[k-1]][obj_key]
            next_val = pop[front_sorted[k+1]][obj_key]
            pop[front_sorted[k]]["crowding"] += (next_val - prev_val) / (vmax - vmin)

def assign_pareto_metrics(pop):
    for ind in pop:
        ind["rank"] = int(1e9)
        ind["crowding"] = 0.0
    fronts = fast_non_dominated_sort(pop, OBJECTIVES)
    for f in fronts:
        crowding_distance(f, pop, OBJECTIVES)
    return fronts

def selection_nsga2(population, target_size):
    fronts = assign_pareto_metrics(population)
    selected = []
    for f in fronts:
        if len(selected) + len(f) <= target_size:
            selected.extend([population[i] for i in f])
        else:
            rest = [population[i] for i in f]
            rest.sort(key=lambda ind: ind["crowding"], reverse=True)
            selected.extend(rest[:target_size - len(selected)])
            break
    return selected

def pareto_tournament_selection(population, tournament_size=2):
    competitors = random.sample(population, tournament_size)
    competitors.sort(key=lambda ind: (ind.get("rank", 1e9), -ind.get("crowding", 0.0)))
    return competitors[0]

# ========== SUPPORTO INDUZIONE (LLM) =========================================

def _collect_history_records(pop_list, limit, objectives):
    """
    Converte lista individui -> record compatti per il prompt.
    Ordina per rank asc, crowding desc, somma obiettivi asc.
    """
    def _obj_tuple(ind):
        return tuple(float(ind.get(k, float('inf'))) for k in objectives)

    items = []
    for ind in pop_list:
        if all(math.isfinite(ind.get(k, float('inf'))) for k in objectives):
            items.append({
                "alleles": [float(a) for a in ind.get("alleli", [])],
                "objectives": {k: float(ind[k]) for k in objectives},
                "rank": int(ind.get("rank", 999999)),
                "crowding": float(ind.get("crowding", 0.0))
            })
    items.sort(key=lambda r: (r["rank"], -r["crowding"], sum(r["objectives"].values())))
    return items[:limit]

def _angles_valid_for_geni(angles_deg, geni, step_deg):
    if len(angles_deg) != len(geni):
        return False
    for ang in angles_deg:
        if not isinstance(ang, (int, float)):
            return False
        if not (0.0 <= (ang % 360.0) < 360.000001):
            return False
        if step_deg and step_deg > 0:
            r = ((ang % 360.0) / step_deg)
            if abs(r - round(r)) > 1e-6:
                return False
    return True

def _snap_to_grid(angles_deg, step_deg):
    if step_deg and step_deg > 0:
        return [(_wrap360(round((a % 360.0)/step_deg)*step_deg)) for a in angles_deg]
    return [_wrap360(a) for a in angles_deg]

def _prompt_for_induction(history_records, geni, step_deg):
    schema = {
        "type": "object",
        "properties": {
            "proposal_degrees": {
                "type": "array",
                "items": {"type": "number"},
                "description": f"Lista di {len(geni)} angoli in gradi in [0,360), multipli di {step_deg}."
            }
        },
        "required": ["proposal_degrees"],
        "additionalProperties": False
    }
    # JSON compatti per ridurre contesto
    history_json = json.dumps(history_records, ensure_ascii=False, separators=(",", ":"))
    schema_json  = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))

    # Istruzione secca: SOLO JSON, nessun testo extra
    msg = (
        f'OUTPUT SOLO JSON, senza testo aggiuntivo.\n'
        f'Numero di geni: {len(geni)}. Ogni valore: gradi in [0,360), multiplo di {step_deg}.\n'
        f'Obiettivi da minimizzare nell’ordine: {OBJECTIVES}.\n'
        f'Storico (compresso): {history_json}\n'
        f'Schema richiesto: {schema_json}\n'
        f'Restituisci esclusivamente un oggetto JSON con la chiave "proposal_degrees". Prova a indurre dai dati storici delle osservazioni quali combinazioni di angoli (geni) minimizzano un obiettivo da te scelto tra quelli da minimizzare.'
    )
    return msg

def _prompt_for_report(history_records, proposal, step_deg):
    """
    Prompt breve per ottenere un report testuale in poche righe.
    Tenere corto per evitare truncation.
    """
    # compatta anche qui
    hist_json = json.dumps(history_records[:20], ensure_ascii=False, separators=(",", ":"))
    prop_json = json.dumps({"proposal_degrees": proposal}, ensure_ascii=False, separators=(",", ":"))

    return (
        "Scrivi un breve report (max 6 righe) sul perché la proposta di angoli è plausibile.\n"
        f"Obiettivi (da minimizzare, in ordine): {OBJECTIVES}\n"
        f"Step angolare: multipli di {step_deg}°.\n"
        "Sottolinea sulla base dello storico quale obiettivo si vuole minimizzare con questa scelta di angoli e le motivazioni dietro tale scelta, visto che la scelta di angoli è stata indotta dalle osservazioni storiche dei dati.\n"
        f"Storico (campione): {hist_json}\n"
        f"Proposta: {prop_json}\n"
        "Rispondi in testo semplice, niente JSON."
    )
    
def _parse_proposal_json(text):
    try:
        obj = json.loads(text)
        arr = obj.get("proposal_degrees", None)
        if isinstance(arr, list) and all(isinstance(x,(int,float)) for x in arr):
            return [float(x) for x in arr]
    except Exception:
        return None
    return None

def _ollama_generate(model, prompt, timeout_s=INDUZIONE_TIMEOUT_S, as_json=False):
    """
    Chiamata generica a Ollama.
    - as_json=True -> forza output JSON puro (stringa JSON in data['response'])
    - ritorna (text, err) dove err è None se ok
    """
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
            "num_ctx": 8192
        }
    }
    if as_json:
        payload["format"] = "json"  # forza JSON puro

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return data.get("response", ""), None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        return None, f"http_error:{type(e).__name__}"
    except Exception as e:
        return None, f"other_error:{type(e).__name__}"
        
        
#def _ollama_chat_generate(model, prompt, timeout_s=INDUZIONE_TIMEOUT_S):
#    url = "http://127.0.0.1:11434/api/generate"
#    payload = {
#        "model": model,
#        "prompt": prompt,
#        "stream": False,
#        "format": "json",  # <<< forza output JSON puro
#        "options": {
#            "temperature": 0.1,
#            "num_predict": 512,
#            "num_ctx": 8192
#        }
#    }
#    try:
#        r = requests.post(url, json=payload, timeout=timeout_s)
#        r.raise_for_status()
#        data = r.json()
#        # In modalità format=json, data["response"] dovrebbe essere una stringa JSON benformata
#        return data.get("response", ""), None
#    except requests.exceptions.Timeout:
#        return None, "timeout"
#    except requests.exceptions.RequestException as e:
#        return None, f"http_error:{type(e).__name__}"
#    except Exception as e:
#        return None, f"other_error:{type(e).__name__}"

#def propose_alleles_by_induction(geni, population_snapshot, model_name, history_k, step_deg, timeout_s, #reask=INDUZIONE_MAX_REASK):
#    history = _collect_history_records(population_snapshot, history_k, OBJECTIVES)
#    if not history:
#        return None, "empty_history"
#    prompt = _prompt_for_induction(history, geni, step_deg)
#    last_err = None
#    for _ in range(max(1, reask+1)):
#        text, err = _ollama_chat_generate(model_name, prompt, timeout_s=timeout_s)
#        if err:
#            last_err = err
#            try:
#                with open("induction_debug.txt", "a") as df:
#                    df.write(f"\n--- gen debug ---\nERR={err}\n{text[:300] if text else ''}\n")
#            except Exception:
#                pass
#            continue
#        angles = _extract_angles_from_text(text)
#        if angles is None:
#            last_err = "bad_json"
#            continue
#        angles = _snap_to_grid(angles, step_deg)
#        if _angles_valid_for_geni(angles, geni, step_deg):
#            return angles, "ok"
#        last_err = "invalid_angles"
#    return None, (last_err or "unknown")

def propose_alleles_by_induction(geni,
                                 population_snapshot,
                                 model_name,
                                 history_k,
                                 step_deg,
                                 timeout_s,
                                 report_path=None,
                                 gen_idx=None,
                                 reask=INDUZIONE_MAX_REASK):
    """
    1) Prima chiamata: chiede SOLO JSON con proposal_degrees (format=json).
    2) Se valida, seconda chiamata: chiede un report testuale e lo salva (best effort).
    Ritorna (angles or None, reason:str)
    """
    history = _collect_history_records(population_snapshot, history_k, OBJECTIVES)
    if not history:
        return None, "empty_history"

    # --- 1) PROPOSTA JSON ---
    prompt1 = _prompt_for_induction(history, geni, step_deg)
    last_err = None
    angles = None
    for _ in range(max(1, reask+1)):
        text, err = _ollama_generate(model_name, prompt1, timeout_s=timeout_s, as_json=True)
        if err:
            last_err = err
            continue
        proposal = _parse_proposal_json(text)
        if proposal is None:
            last_err = "bad_json"
            continue
        proposal = _snap_to_grid(proposal, step_deg)
        if _angles_valid_for_geni(proposal, geni, step_deg):
            angles = proposal
            break
        last_err = "invalid_angles"

    if angles is None:
        return None, (last_err or "unknown")

    # --- 2) REPORT TESTUALE (best effort, non blocca) ---
    if report_path:
        try:
            prompt2 = _prompt_for_report(history, angles, step_deg)
            report_text, err2 = _ollama_generate(model_name, prompt2, timeout_s=min(timeout_s, 15.0), as_json=False)
            if report_text:
                with open(report_path, "a") as rf:
                    tag = f"[gen={gen_idx}]" if gen_idx is not None else ""
                    rf.write(f"\n=== Induction report {tag} ===\n{report_text.strip()}\n")
            else:
                # log minimal in mancanza di testo
                with open(report_path, "a") as rf:
                    tag = f"[gen={gen_idx}]" if gen_idx is not None else ""
                    rf.write(f"\n=== Induction report {tag} ===\n<report non disponibile: {err2}>\n")
        except Exception:
            pass  # mai bloccare il GA per il report

    return angles, "ok"

def _extract_angles_from_text(text):
    # 1) prova JSON diretto
    try:
        obj = json.loads(text)
        arr = obj.get("proposal_degrees", None)
        if isinstance(arr, list) and all(isinstance(x,(int,float)) for x in arr):
            return [float(x) for x in arr]
    except Exception:
        pass

    # 2) prova a trovare direttamente un array numerica [...]
    m = re.search(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*?)\s*\]", text)
    if m:
        try:
            arr_txt = "[" + m.group(1) + "]"
            arr = json.loads(arr_txt)
            if isinstance(arr, list) and all(isinstance(x,(int,float)) for x in arr):
                return [float(x) for x in arr]
        except Exception:
            pass

    # 3) blocco ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            arr = obj.get("proposal_degrees", None)
            if isinstance(arr, list) and all(isinstance(x,(int,float)) for x in arr):
                return [float(x) for x in arr]
        except Exception:
            pass

    return None


# ========== POPOLAZIONE & VALUTAZIONE =======================================

def initialize_population(pop_size, geni):
    population = []
    for i in range(pop_size):
        alleli = [generate_random_allele(period) for period, _ in geni]
        population.append({
            "id": i,
            "alleli": alleli,
            # multi-obiettivo (inizializza a +inf)
            "fitness_hbond": float("inf"),
            "fitness_hb_bifork": float("inf"),
            "fitness_energy": float("inf"),
            "fitness_grms": float("inf"),
            "fitness_gmax": float("inf"),
            "rank": None,
            "crowding": 0.0,
            # output:
            "xyz_file": None,
            "num_atoms": None,
            "rotA": None, "rotB": None, "rotC": None,
            "xyz_lines": [],
            # info:
            "helped": False,
            "hb_details": {}
        })
    return population

def evaluate_individual(ind, geni, tmp_dir, gen_dir,
                        initial_bonds=None,
                        donors=None, donors_H=None, acceptors=None,
                        hb_sphere=HB_SPHERE,
                        hb_bonus=HB_BONUS_PER_BOND,
                        hb_mutual_penalty=HB_MUTUAL_PENALTY,
                        use_standard=USE_STANDARD_ORIENTATION,
                        max_topology_tries=5,
                        rescue_pool=None):
    base_content = strip_gene_lines(INPUT_FILE)
    base_lines = remove_frozen_substring(base_content.splitlines(True))

    tries = 0
    while True:
        tries += 1
        individual_lines = add_gene_lines(base_lines, geni, ind["alleli"])
        gjf_file = write_individual_file(tmp_dir, ind["id"], individual_lines)
        log_file = os.path.join(tmp_dir, f"individuo_{ind['id']}.log")

        retcode = run_gdv(gjf_file, log_file)
        if retcode != 0:
            cleanup_individual_tmp(tmp_dir, ind["id"])
            ind["alleli"], _ = tweak_some_alleles_random(ind["alleli"], geni)
            if tries >= max_topology_tries:
                if rescue_pool is not None and clone_from_rescue(ind, rescue_pool):
                    return
                _invalidate(ind)
                return
            continue

        # --- PARSE DEI 4 OBIETTIVI ---
        energy = parse_fitness(log_file)               # (Hartree)
        grms, gmax = parse_cartesian_forces_rms(log_file)  # (force units from log)

        if (energy is None) or (grms is None) or (gmax is None) or \
           (not math.isfinite(energy)) or (not math.isfinite(grms)) or (not math.isfinite(gmax)):
            cleanup_individual_tmp(tmp_dir, ind["id"])
            ind["alleli"], _ = tweak_some_alleles_random(ind["alleli"], geni)
            if tries >= max_topology_tries:
                if rescue_pool is not None and clone_from_rescue(ind, rescue_pool):
                    return
                _invalidate(ind)
                return
            continue

        coords = parse_last_orientation_coords(log_file, use_standard=use_standard)
        if not coords:
            coords = parse_last_orientation_coords(log_file, use_standard=(not use_standard))

        if initial_bonds is not None and coords:
            current_bonds = build_bond_graph(coords)
            if (len(current_bonds) != len(initial_bonds)) or (not same_topology(current_bonds, initial_bonds)):
                cleanup_individual_tmp(tmp_dir, ind["id"])
                ind["alleli"], _ = tweak_some_alleles_random(ind["alleli"], geni)
                if tries >= max_topology_tries:
                    if rescue_pool is not None and clone_from_rescue(ind, rescue_pool):
                        return
                    _invalidate(ind)
                    return
                continue
        else:
            if rescue_pool is not None and clone_from_rescue(ind, rescue_pool):
                return
            _invalidate(ind)
            return

        # --- Fitness HB ---
        rotA, rotB, rotC = parse_rotational_constants_mhz(log_file)
        ind["rotA"] = rotA; ind["rotB"] = rotB; ind["rotC"] = rotC

        hb_fit, hb_det, helped = evaluate_hbond_fitness(
            coords, donors, donors_H, acceptors,
            sphere=hb_sphere, bonus=hb_bonus, mutual_penalty=hb_mutual_penalty,
            bonds=initial_bonds
        )
        
        # --- Fitness H-bond biforcati ---
        hb_bif_fit, hb_bif_det = evaluate_bifurcated_hbond_fitness(
            coords, donors, donors_H, acceptors,
            contact_threshold=HB_CONTACT_THRESHOLD,
            bonds=initial_bonds
        )
        ind["fitness_hb_bifork"] = hb_bif_fit
        if "hb_details" in ind and isinstance(ind["hb_details"], dict):
            ind["hb_details"]["bifork"] = hb_bif_det
        else:
            ind["hb_details"] = {"bifork": hb_bif_det}

        # assegna tutte le fitness
        ind["fitness_energy"] = energy
        ind["fitness_grms"]   = grms
        ind["fitness_gmax"]   = gmax
        ind["fitness_hbond"]  = hb_fit
        ind["hb_details"]     = hb_det
        ind["helped"]         = helped

        num_atoms, xyz_lines = parse_xyz_from_log(log_file)
        ind["num_atoms"] = num_atoms
        ind["xyz_lines"] = xyz_lines
        ind["xyz_file"]  = None

        if rescue_pool is not None and all(math.isfinite(ind[k]) for k in OBJECTIVES):
            rescue_pool.append(snapshot_for_rescue(ind))
        return

def _invalidate(ind):
    for k in OBJECTIVES:
        ind[k] = float("inf")
    ind["fitness_hb_bifork"] = float("inf")
    ind["num_atoms"] = None
    ind["xyz_lines"] = []
    ind["xyz_file"]  = None

# ========== CLI ==============================================================

def parse_pairs_arg(pairs_str):
    if not pairs_str:
        return []
    items = re.split(r'\s*,\s*', pairs_str.strip())
    pairs = []
    for it in items:
        m = re.match(r'^\s*(\d+)\s*[-:]\s*(\d+)\s*$', it)
        if not m:
            raise ValueError(f"Formato coppia non valido: '{it}'. Usa es. '1-5,2-7'")
        i, j = int(m.group(1)), int(m.group(2))
        pairs.append((i, j))
    return pairs

def parse_weights_arg(weights_str):
    if not weights_str:
        return []
    items = re.split(r'\s*,\s*', weights_str.strip())
    ws = []
    for it in items:
        try:
            ws.append(float(it))
        except ValueError:
            raise ValueError(f"Peso non numerico: '{it}'")
    return ws

def build_arg_parser():
    parser = argparse.ArgumentParser(description="GA conformer search (multi-obiettivo: HB + Energy + GRMS + GMAX).")
    parser.add_argument("--gjf", type=str, default=INPUT_FILE, help="Percorso al file .gjf di input.")
    parser.add_argument("--out-dir", type=str, default=GENERATIONS_DIR, help="Directory output delle generazioni.")
    parser.add_argument("--tmp-dir", type=str, default=TMP_DIR, help="Directory temporanea per run intermedi.")
    parser.add_argument("--seed", type=int, default=None, help="Seed RNG (default: None).")
    parser.add_argument("--num-generazioni", type=int, default=NUM_GENERAZIONI)
    parser.add_argument("--pop-iniziale", type=int, default=POPOLAZIONE_INIZIALE)
    parser.add_argument("--pop-target", type=int, default=POPOLAZIONE_TARGET)
    parser.add_argument("--cpu-fraction", type=float, default=0.75)
    parser.add_argument("--gene-sim-threshold-deg", type=float, default=5.0)
    parser.add_argument("--near-pairs", type=str, default="")
    parser.add_argument("--near-weights", type=str, default="")
    parser.add_argument("--pair-sphere", type=float, default=PAIR_SPHERE)
    parser.add_argument("--use-standard-orientation", action="store_true")
    parser.add_argument("--use-principal-axis", action="store_true")
    parser.add_argument("--max-topology-tries", type=int, default=20,
                        help="Numero massimo di tentativi di rigenerazione alleli se la topologia cambia.")
    parser.add_argument("--hb-sphere", type=float, default=HB_SPHERE, help="Raggio H...A per H-bond (Å).")
    parser.add_argument("--hb-bonus", type=float, default=HB_BONUS_PER_BOND, help="Quanto riduce la seconda fitness per H-bond.")
    parser.add_argument("--hb-mutual-penalty", type=float, default=HB_MUTUAL_PENALTY, help="Penalità per coppie reciproche A↔B.")
    parser.add_argument("--sbx-eta", type=float, default=SBX_ETA, help="Indice di distribuzione SBX.")

    # === NUOVE FLAG PER INDUZIONE (LLM) ===
    parser.add_argument("--p-induzione", type=float, default=P_INDUZIONE_DEFAULT,
                        help="Probabilità di usare l'operatore 'induzione' (LLM).")
    parser.add_argument("--induzione-model", type=str, default=INDUZIONE_MODEL_DEFAULT,
                        help="Nome modello Ollama (es. 'mistral:7b-instruct').")
    parser.add_argument("--induzione-timeout", type=float, default=INDUZIONE_TIMEOUT_S,
                        help="Timeout in secondi per la chiamata al modello locale.")
    parser.add_argument("--induzione-history-k", type=int, default=INDUZIONE_HISTORY_K,
                        help="Massimo numero di individui di storico da passare.")
    parser.add_argument("--induzione-step-deg", type=float, default=INDUZIONE_STEP_DEG,
                        help="Passo di discretizzazione in gradi richiesto al modello.")
    # ======================================
    return parser

# ========== MAIN GA ==========================================================

def genetic_algorithm():
    global INPUT_FILE, TMP_DIR, GENERATIONS_DIR
    global NUM_GENERAZIONI, POPOLAZIONE_INIZIALE, POPOLAZIONE_TARGET
    global GENI, USE_STANDARD_ORIENTATION
    global HB_SPHERE, HB_BONUS_PER_BOND, HB_MUTUAL_PENALTY, SBX_ETA

    parser = build_arg_parser()
    try:
        args = parser.parse_args()
    except SystemExit:
        class _Args: pass
        args = _Args()
        args.gjf = INPUT_FILE
        args.out_dir = GENERATIONS_DIR
        args.tmp_dir = TMP_DIR
        args.num_generazioni = NUM_GENERAZIONI
        args.pop_iniziale = POPOLAZIONE_INIZIALE
        args.pop_target = POPOLAZIONE_TARGET
        args.cpu_fraction = 0.75
        args.gene_sim_threshold_deg = 5.0
        args.near_pairs = ""
        args.near_weights = ""
        args.pair_sphere = PAIR_SPHERE
        args.use_standard_orientation = True
        args.use_principal_axis = False
        args.hb_sphere = HB_SPHERE
        args.hb_bonus = HB_BONUS_PER_BOND
        args.hb_mutual_penalty = HB_MUTUAL_PENALTY
        args.sbx_eta = SBX_ETA
        # default induzione
        args.p_induzione = P_INDUZIONE_DEFAULT
        args.induzione_model = INDUZIONE_MODEL_DEFAULT
        args.induzione_timeout = INDUZIONE_TIMEOUT_S
        args.induzione_history_k = INDUZIONE_HISTORY_K
        args.induzione_step_deg = INDUZIONE_STEP_DEG

    MAX_TOPOLOGY_TRIES = int(getattr(args, "max_topology_tries", 10))
    INPUT_FILE = args.gjf
    TMP_DIR = args.tmp_dir
    GENERATIONS_DIR = args.out_dir
    NUM_GENERAZIONI = int(args.num_generazioni)
    POPOLAZIONE_INIZIALE = int(args.pop_iniziale)
    POPOLAZIONE_TARGET = int(args.pop_target)
    if args.seed is not None:
        random.seed(args.seed)

    USE_STANDARD_ORIENTATION = False
    if getattr(args, "use_principal_axis", False):
        USE_STANDARD_ORIENTATION = False
    elif getattr(args, "use_standard_orientation", False):
        USE_STANDARD_ORIENTATION = True

    HB_SPHERE = float(getattr(args, "hb_sphere", HB_SPHERE))
    HB_BONUS_PER_BOND = float(getattr(args, "hb_bonus", HB_BONUS_PER_BOND))
    HB_MUTUAL_PENALTY = float(getattr(args, "hb_mutual_penalty", HB_MUTUAL_PENALTY))
    SBX_ETA = float(getattr(args, "sbx_eta", SBX_ETA))

    # Parametri induzione
    P_INDUZIONE = float(getattr(args, "p_induzione", P_INDUZIONE_DEFAULT))
    INDUZIONE_MODEL = getattr(args, "induzione_model", INDUZIONE_MODEL_DEFAULT)
    INDUZIONE_TOUT = float(getattr(args, "induzione_timeout", INDUZIONE_TIMEOUT_S))
    INDUZIONE_HK   = int(getattr(args, "induzione_history_k", INDUZIONE_HISTORY_K))
    INDUZIONE_STEP = float(getattr(args, "induzione_step_deg", INDUZIONE_STEP_DEG))

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(GENERATIONS_DIR, exist_ok=True)

    generations_data = []
    generation_log_path = os.path.join(GENERATIONS_DIR, "generation_log.txt")
    with open(generation_log_path, "w") as logf:
        logf.write("Log generazioni (multi-obiettivo: HB + HBbif + Energy + GRMS + GMAX):\n\n")
        logf.write(f"[INDUZIONE] p={P_INDUZIONE}  model={INDUZIONE_MODEL}  timeout={INDUZIONE_TOUT}s  K={INDUZIONE_HK}  step={INDUZIONE_STEP}\n\n")
        
    induction_report_path = os.path.join(GENERATIONS_DIR, "induction_reports.txt")
    # opzionale: init file
    try:
        with open(induction_report_path, "w") as rf:
            rf.write("Report induzione (LLM)\n")
    except Exception:
        induction_report_path = None  # in caso non sia scrivibile
        
        
    parsed_geni = parse_genes_from_gjf(INPUT_FILE)
    if parsed_geni:
        GENI = parsed_geni

    reference_lines = read_reference_file(INPUT_FILE)

    _initial_coords = parse_coords_from_gjf(INPUT_FILE)
    _initial_bonds = build_bond_graph(_initial_coords) if _initial_coords else None

    donors, donors_H, acceptors = identify_donors_acceptors(_initial_coords, _initial_bonds) if _initial_coords and _initial_bonds else ([], {}, [])

    population = initialize_population(POPOLAZIONE_INIZIALE, GENI)
    rescue_pool = []
    experience_buffer = []  # buffer di esperienza per l’operatore di induzione

    # log opzionale uso induzione
    induction_log_path = os.path.join(GENERATIONS_DIR, "induction_log.txt")
    with open(induction_log_path, "w") as lf:
        lf.write("Log proposte LLM (induzione)\n")

    for gen in range(NUM_GENERAZIONI):
        print(f"Generazione {gen}")

        phase = (math.pi * NUM_OSCILLATIONS / (max(1, NUM_GENERAZIONI - 1))) * gen
        current_mutation_rate = BASE_RATE_MUTATION + DELTA_RATE_MUTATION * math.sin(phase)
        current_crossover_rate = BASE_RATE_CROSSOVER - DELTA_RATE_CROSSOVER * math.sin(phase)

        gen_dir = os.path.join(GENERATIONS_DIR, f"population_{gen}")
        os.makedirs(gen_dir, exist_ok=True)

        max_workers = max(1, int((os.cpu_count() or 1) * max(0.01, min(1.0, args.cpu_fraction))))
        max_workers = min(max_workers, len(population))

        # Valutazione parallela
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_individual,
                    ind, GENI, TMP_DIR, gen_dir,
                    _initial_bonds, donors, donors_H, acceptors,
                    HB_SPHERE, HB_BONUS_PER_BOND, HB_MUTUAL_PENALTY,
                    USE_STANDARD_ORIENTATION,
                    MAX_TOPOLOGY_TRIES,
                    rescue_pool
                )
                for ind in population
            ]
            for future in as_completed(futures):
                future.result()

        # Selezione NSGA-II
        selected = selection_nsga2(population, POPOLAZIONE_TARGET)

        # Aggiorna esperienza
        experience_buffer.extend(copy.deepcopy(selected))
        max_store = int(getattr(args, "induzione_history_k", INDUZIONE_HISTORY_K)) * 5
        if len(experience_buffer) > max_store:
            experience_buffer = experience_buffer[-max_store:]

        # Statistiche
        HB_list     = [ind["fitness_hbond"]    for ind in selected if math.isfinite(ind["fitness_hbond"])]
        HB_BIF_list = [ind["fitness_hb_bifork"]for ind in selected if math.isfinite(ind["fitness_hb_bifork"])]
        E_list      = [ind["fitness_energy"]   for ind in selected if math.isfinite(ind["fitness_energy"])]
        GRMS_list   = [ind["fitness_grms"]     for ind in selected if math.isfinite(ind["fitness_grms"])]
        GMAX_list   = [ind["fitness_gmax"]     for ind in selected if math.isfinite(ind["fitness_gmax"])]

        def _stats(L):
            if not L: return (None, None, None)
            return (sum(L)/len(L), max(L), min(L))

        avg_HB, max_HB, min_HB             = _stats(HB_list)
        avg_HB_BIF, max_HB_BIF, min_HB_BIF = _stats(HB_BIF_list)
        avg_E,  max_E,  min_E              = _stats(E_list)
        avg_G,  max_G,  min_G              = _stats(GRMS_list)
        avg_M,  max_M,  min_M              = _stats(GMAX_list)

        with open(generation_log_path, "a") as logf:
            logf.write(f"Generazione {gen}:\n")
            logf.write(f"  HB      -> avg: {avg_HB} | min: {min_HB} | max: {max_HB}\n")
            logf.write(f"  HBbif   -> avg: {avg_HB_BIF} | min: {min_HB_BIF} | max: {max_HB_BIF}\n")
            logf.write(f"  Energy  -> avg: {avg_E}  | min: {min_E}  | max: {max_E}\n")
            logf.write(f"  GRMS    -> avg: {avg_G}  | min: {min_G}  | max: {max_G}\n")
            logf.write(f"  GMAX    -> avg: {avg_M}  | min: {min_M}  | max: {max_M}\n")
            logf.write(f"  Mutation rate: {current_mutation_rate}\n")
            logf.write(f"  Crossover rate: {current_crossover_rate}\n")
            logf.write("  Individui target (rank, crowding, HB, HBbif, E, GRMS, GMAX):\n")
            for ind in selected:
                rotA = ind.get('rotA'); rotB = ind.get('rotB'); rotC = ind.get('rotC')
                extra_bif = f" | HBbif={ind['fitness_hb_bifork']}" if "fitness_hb_bifork" in ind else ""
                logf.write(
                    f"    id={ind['id']} | rank={ind.get('rank')} | crowd={ind.get('crowding')} "
                    f"| HB={ind['fitness_hbond']}{extra_bif} | E={ind['fitness_energy']} | GRMS={ind['fitness_grms']} | GMAX={ind['fitness_gmax']}"
                )
                if rotA is not None and rotB is not None and rotC is not None:
                    logf.write(f" | A={rotA} | B={rotB} | C={rotC}")
                if ind.get("helped"):
                    logf.write(" [HBOND]\n")
                else:
                    logf.write("\n")
            logf.write("\n")

        generations_data.append({
            "Generation": gen,
            "Avg. HB": avg_HB, "MAX HB": max_HB, "MIN HB": min_HB,
            "Avg. HB Bif.": avg_HB_BIF, "MAX HB Bif.": max_HB_BIF, "MIN HB Bif.": min_HB_BIF,
            "Avg. Energy": avg_E, "MAX Energy": max_E, "MIN Energy": min_E,
            "Avg. GRMS": avg_G, "MAX GRMS": max_G, "MIN GRMS": min_G,
            "Avg. GMAX": avg_M, "MAX GMAX": max_M, "MIN GMAX": min_M,
            "Mutation rate": current_mutation_rate,
            "Crossover rate": current_crossover_rate
        })

        # Salva XYZ dei selezionati
        for ind in selected:
            if ind.get("xyz_lines"):
                n_atoms = ind.get("num_atoms") or len(ind["xyz_lines"])
                ind["xyz_file"] = write_xyz_file(
                    gen_dir, ind["id"],
                    ind["fitness_hbond"], ind["fitness_energy"], ind["fitness_grms"], ind["fitness_gmax"],
                    ind.get("rank"), n_atoms, ind["xyz_lines"],
                    rotA=ind.get("rotA"), rotB=ind.get("rotB"), rotC=ind.get("rotC"),
                    hb_bif=ind.get("fitness_hb_bifork")
                )

        assign_pareto_metrics(selected)

        # === Riproduzione: aggiunta operatore di INDUZIONE ===
        new_population = []
        while len(new_population) < POPOLAZIONE_INIZIALE:
            parent1 = pareto_tournament_selection(selected, tournament_size=2)
            parent2 = pareto_tournament_selection(selected, tournament_size=2)

            # usa l'induzione SOLO dalla metà evoluzione in poi
            use_induction = (gen >= (NUM_GENERAZIONI // 2)) and (random.random() < P_INDUZIONE)
            if use_induction:
                #seed_pool = experience_buffer if experience_buffer else selected
                #proposal, reason = propose_alleles_by_induction(
                #    GENI, seed_pool, INDUZIONE_MODEL, INDUZIONE_HK, INDUZIONE_STEP, INDUZIONE_TOUT
                #)
                seed_pool = experience_buffer if experience_buffer else selected
                proposal, reason = propose_alleles_by_induction(
                    GENI, seed_pool,
                    INDUZIONE_MODEL, INDUZIONE_HK,
                    INDUZIONE_STEP, INDUZIONE_TOUT,
                    report_path=induction_report_path,   # <— salva il report qui
                    gen_idx=gen,                         # <— utile per etichettare
                    reask=INDUZIONE_MAX_REASK
                )
                if proposal is not None:
                    child_alleles = proposal
                    try:
                        with open(induction_log_path, "a") as lf:
                            lf.write(f"gen={gen}  use_induction=1  result=ok  proposal={child_alleles}\n")
                    except Exception:
                        pass
                else:
                    # fallback standard
                    if random.random() < current_crossover_rate:
                        child_alleles = crossover_sbx(parent1, parent2, eta=SBX_ETA)
                    else:
                        child_alleles = parent1['alleli'].copy()
                    child_alleles = mutate(child_alleles, current_mutation_rate, GENI)
                    try:
                        with open(induction_log_path, "a") as lf:
                            lf.write(f"gen={gen}  use_induction=1  result=fail  reason={reason}  fallback=1\n")
                    except Exception:
                        pass
            else:
                # percorso standard
                if random.random() < current_crossover_rate:
                    child_alleles = crossover_sbx(parent1, parent2, eta=SBX_ETA)
                else:
                    child_alleles = parent1["alleli"].copy()
                child_alleles = mutate(child_alleles, current_mutation_rate, GENI)
                try:
                    with open(induction_log_path, "a") as lf:
                        lf.write(f"gen={gen}  use_induction=0  reason=guard_or_prob  fallback=1\n")
                except Exception:
                    pass

            # evita duplicati (genotipi troppo simili nella nuova popolazione)
            if any(are_genotypes_similar(child_alleles, ind["alleli"], args.gene_sim_threshold_deg) for ind in new_population):
                child_alleles, _ = tweak_some_alleles_random(child_alleles, GENI, k=1)

            new_population.append({
                "id": random.randint(1000, 9999),
                "alleli": child_alleles,
                "fitness_hbond": float("inf"),
                "fitness_hb_bifork": float("inf"),
                "fitness_energy": float("inf"),
                "fitness_grms": float("inf"),
                "fitness_gmax": float("inf"),
                "rank": None,
                "crowding": 0.0,
                "xyz_file": None,
                "rotA": None, "rotB": None, "rotC": None,
                "num_atoms": None,
                "xyz_lines": [],
                "helped": False,
                "hb_details": {}
            })
        population = new_population
        cleanup_tmp(TMP_DIR)

    # --- Dump cumulativi ---
    cum_helped_dir = os.path.join(GENERATIONS_DIR, "cumulative_helped")
    cum_standard_dir = os.path.join(GENERATIONS_DIR, "cumulative_standard")
    os.makedirs(cum_helped_dir, exist_ok=True)
    os.makedirs(cum_standard_dir, exist_ok=True)
    
    def _dominates(a, b, keys):
        return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)

    def global_pareto_front_from_selected(rootdir):
        items = []
        for root, _, files in os.walk(rootdir):
            if "population_" not in root:
                continue
            for fn in files:
                if not fn.endswith(".xyz"):
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, "r") as f:
                        _ = f.readline()
                        meta = f.readline().strip()
                    mHB = re.search(r"\bHB=([-\d\.Ee+]+)", meta)
                    mE  = re.search(r"\bE=([-\d\.Ee+]+)", meta)
                    mG  = re.search(r"\bGRMS=([-\d\.Ee+]+)", meta)
                    mM  = re.search(r"\bGMAX=([-\d\.Ee+]+)", meta)
                    mBf = re.search(r"\bHBbif=([-\d\.Ee+]+)", meta)

                    rec = {
                        "fitness_hbond":  float(mHB.group(1)) if mHB else float("inf"),
                        "fitness_energy": float(mE.group(1))  if mE  else float("inf"),
                        "fitness_grms":   float(mG.group(1))  if mG  else float("inf"),
                        "fitness_gmax":   float(mM.group(1))  if mM  else float("inf"),
                        "path": path
                    }
                    if mBf:
                        rec["fitness_hb_bifork"] = float(mBf.group(1))
                    items.append(rec)
                except:
                    continue

        obj_keys = ["fitness_hbond","fitness_energy","fitness_grms","fitness_gmax"]
        if any("fitness_hb_bifork" in r for r in items):
            obj_keys.insert(1, "fitness_hb_bifork")

        front = []
        for a in items:
            dominated = False
            for b in items:
                if a is b:
                    continue
                if _dominates(b, a, obj_keys):
                    dominated = True
                    break
            if not dominated:
                front.append(a)
        return front

    def dump_global_front(front, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "global_pareto_front.csv")
        has_bif = any("fitness_hb_bifork" in r for r in front)
        with open(log_path, "w", newline="") as cf:
            w = csv.writer(cf)
            header = ["HB","E","GRMS","GMAX","file"]
            if has_bif:
                header.insert(1, "HBbif")
            w.writerow(header)
            def _key(r):
                base = (r["fitness_hbond"],)
                if has_bif:
                    base += (r["fitness_hb_bifork"],)
                base += (r["fitness_energy"], r["fitness_grms"], r["fitness_gmax"])
                return base
            for i, rec in enumerate(sorted(front, key=_key)):
                base = os.path.basename(rec["path"])
                dst = os.path.join(out_dir, f"{os.path.splitext(base)[0]}_PF{i}.xyz")
                shutil.copyfile(rec["path"], dst)
                if has_bif:
                    row = [rec["fitness_hbond"], rec["fitness_hb_bifork"], rec["fitness_energy"], rec["fitness_grms"], rec["fitness_gmax"], os.path.basename(dst)]
                else:
                    row = [rec["fitness_hbond"], rec["fitness_energy"], rec["fitness_grms"], rec["fitness_gmax"], os.path.basename(dst)]
                w.writerow(row)

    def _collect_xyz(rootdir):
        out = []
        for root, _, files in os.walk(rootdir):
            for fn in files:
                if fn.endswith(".xyz") and "population_" in root:
                    path = os.path.join(root, fn)
                    try:
                        with open(path, "r") as f:
                            _ = f.readline()
                            meta = f.readline().strip()
                        mHB = re.search(r"\bHB=([-\d\.Ee+]+)", meta)
                        mE  = re.search(r"\bE=([-\d\.Ee+]+)", meta)
                        mG  = re.search(r"\bGRMS=([-\d\.Ee+]+)", meta)
                        mM  = re.search(r"\bGMAX=([-\d\.Ee+]+)", meta)
                        HB = float(mHB.group(1)) if mHB else float("inf")
                        E  = float(mE.group(1))  if mE  else float("inf")
                        G  = float(mG.group(1))  if mG  else float("inf")
                        M  = float(mM.group(1))  if mM  else float("inf")
                        mA  = re.search(r"\bA=([-\d\.Ee+]+)", meta)
                        mB  = re.search(r"\bB=([-\d\.Ee+]+)", meta)
                        mC  = re.search(r"\bC=([-\d\.Ee+]+)", meta)
                        A  = float(mA.group(1))  if mA  else None
                        B  = float(mB.group(1))  if mB  else None
                        C  = float(mC.group(1))  if mC  else None
                    except:
                        HB = E = G = M = float("inf"); A = B = C = None
                    out.append((HB, E, G, M, A, B, C, path))
        return out

    all_xyz = _collect_xyz(GENERATIONS_DIR)
    helped   = [t for t in all_xyz if t[0] < -0.3]   # HB < -0.3
    standard = [t for t in all_xyz if not (t[0] < -0.3)]

    helped.sort(key=lambda t: (t[0], t[1], t[2], t[3]))     # HB, E, GRMS, GMAX
    standard.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    def _dump_cum(lst, out_dir, log_name):
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, log_name)
        with open(log_path, "w") as lf:
            lf.write(f"Elenco (ordinato per HB, E, GRMS, GMAX) - n={len(lst)}\n")
            for rank, (HB, E, G, M, A, B, C, src) in enumerate(lst):
                base = os.path.basename(src)
                root, ext = os.path.splitext(base)
                dst_name = f"{root}_{rank}{ext}"
                dst_path = os.path.join(out_dir, dst_name)
                shutil.copyfile(src, dst_path)
                if A is not None and B is not None and C is not None:
                    lf.write(f"{rank:04d}  HB={HB}  E={E}  GRMS={G}  GMAX={M}  A={A}  B={B}  C={C}  file={dst_name}\n")
                else:
                    lf.write(f"{rank:04d}  HB={HB}  E={E}  GRMS={G}  GMAX={M}  file={dst_name}\n")

    cum_helped_dir = os.path.join(GENERATIONS_DIR, "cumulative_helped")
    cum_standard_dir = os.path.join(GENERATIONS_DIR, "cumulative_standard")
    _dump_cum(helped,   cum_helped_dir,   "cumulative_helped_log.txt")
    _dump_cum(standard, cum_standard_dir, "cumulative_standard_log.txt")

    cum_all_dir = os.path.join(GENERATIONS_DIR, "cumulative")
    os.makedirs(cum_all_dir, exist_ok=True)
    all_sorted = sorted(all_xyz, key=lambda t: (t[0], t[1], t[2], t[3]))
    _dump_cum(all_sorted, cum_all_dir, "cumulative_all_log.txt")

    forces_path = os.path.join(cum_all_dir, "cumulative_all_forces.txt")
    with open(forces_path, "w") as ef:
        ef.write(f"Elenco forze (ordinato per HB, E, GRMS, GMAX) - n={len(all_sorted)}\n")
        for rank, (HB, E, G, M, A, B, C, src) in enumerate(all_sorted):
            base = os.path.basename(src)
            if A is not None and B is not None and C is not None:
                ef.write(f"{rank:04d}  HB={HB}  E={E}  GRMS={G}  GMAX={M}  A={A}  B={B}  C={C}  file={base}\n")
            else:
                ef.write(f"{rank:04d}  HB={HB}  E={E}  GRMS={G}  GMAX={M}  file={base}\n")

    with open(generation_log_path, "a") as logf:
        logf.write("=== Riepilogo finale ===\n")
        logf.write(f"  Totale selezionati HELPED (HB<-0.3): {len(helped)}\n")
        logf.write(f"  Totale selezionati STANDARD:         {len(standard)}\n")
        logf.write(f"  Cartella cumulative_helped:   {cum_helped_dir}\n")
        logf.write(f"  Cartella cumulative_standard: {cum_standard_dir}\n")
        logf.write(f"  Cartella cumulative_all:      {cum_all_dir}\n\n")
        
    # --- Pareto globale su tutti i selezionati (tutte le generazioni) ---
    global_pf = global_pareto_front_from_selected(GENERATIONS_DIR)
    dump_global_front(global_pf, os.path.join(GENERATIONS_DIR, "global_pareto_front"))
    save_statistics(os.path.join(GENERATIONS_DIR, "evolution.csv"), generations_data)
    print("Algoritmo completato.")

if __name__ == "__main__":
    genetic_algorithm()
    

