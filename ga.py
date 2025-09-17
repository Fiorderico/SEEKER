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
from typing import Optional

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

# --- H-bond objective (NUOVO) ---
HB_SPHERE = 2.5               # Å: raggio per considerare H-bond (H...A)
HB_BONUS_PER_BOND = 0.02      # quanto RIDUCE (migliora) la seconda fitness per ciascun H-bond
HB_MUTUAL_PENALTY = 0.05      # penalità se trovi A→B e B→A nella stessa geometria

# --- Pesi diversi per tipologia di H-bond (NUOVO) ---
HB_EPS_BASE = 1.0     # default (ad es. N–H...O e altri)
HB_EPS_OHN  = 1.1     # più profondo per O–H...N (AUMENTA a piacere)
HB_EPS_OHO  = 1.2     # più profondo per O–H...O (AUMENTA a piacere)
HB_EPS_NHO  = 1.0     # esplicito per N–H...O (puoi variare se vuoi)
#TODO: Metti SH la metà dei corrispondenti (N/O)

# --- H–H penalty params (NUOVO) ---
HH_EPS   = 0.1
HH_ALPHA = 15.0
HH_XM    = 2.0  # Å
#TODO: Se tre idrogeni stanno sul triangolo non ce lo metti (ponte a idrogeno non lineare)

# --- SBX (NUOVO) ---
SBX_ETA = 15.0                # indice di distribuzione
ANGLE_LOW = 0.0
ANGLE_HIGH = 360.0

# --- Orientazioni parse ---
USE_STANDARD_ORIENTATION = True

# (Manteniamo, ma NON usiamo più le coppie vicino/lontano)
PAIR_SPHERE = 2.5
NEAR_PAIRS = []
NEAR_WEIGHTS = []

# --- “GENI” letti dal gjf ---
GENI = []

# --- Mappa simbolo -> Z ---
ELEMENT_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "S": 16
    # estendi se serve
}

# --- Raggi covalenti (Å) ---
COV_RADII = {
    1: 0.31,  # H
    6: 0.76,  # C
    7: 0.71,  # N
    8: 0.66,  # O
    16: 1.01  # S
}

# --- Raggi di van der Waals (Å) (Bondi) ---
VDW_RADII = {
    1: 1.20,  # H
    6: 1.70,  # C
    7: 1.55,  # N
    8: 1.52,  # O
    16: 1.80, # S
}

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
    """
    Cambia un numero casuale di geni (k in [1, n]) estraendo nuovi alleli
    coerenti con la periodicità. Evita (per quanto possibile) di ripescare
    esattamente lo stesso valore.
    """
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
    """Copia 'leggera' di quanto serve per duplicare un individuo sano."""
    return {
        "alleli": ind["alleli"][:],
        "fitness_energy": ind["fitness_energy"],
        "fitness_hbond": ind["fitness_hbond"],
        "num_atoms": ind.get("num_atoms"),
        "xyz_lines": ind.get("xyz_lines", [])[:],
        "helped": ind.get("helped", False),
        "hb_details": copy.deepcopy(ind.get("hb_details", {})),
    }

def clone_from_rescue(ind, rescue_pool):
    """
    Se possibile, clona su 'ind' un individuo sano scelto dal rescue_pool.
    Ritorna True se il clone è avvenuto, altrimenti False.
    """
    if not rescue_pool:
        return False
    donor = random.choice(rescue_pool)
    ind["alleli"]         = donor["alleli"][:]
    ind["fitness_energy"] = donor["fitness_energy"]
    ind["fitness_hbond"]  = donor["fitness_hbond"]
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

def parse_fitness(log_file):
    with open(log_file, 'r', errors='ignore') as f:
        content = f.read()
    m = re.search(r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)", content)
    if m:
        return float(m.group(1).replace('D','E'))
    compact = content.replace("\n","").replace(" ","")
    matches = re.findall(r"HF=([^\\]+)\\", compact)
    if matches:
        raw = matches[-1].replace('D','E').replace('d','E')
        try:
            return float(raw)
        except:
            pass
    m = re.search(r"Energy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)", content)
    if m:
        return float(m.group(1).replace('D','E'))
    print(f"Fitness non trovata in {log_file}")
    return None

def parse_xyz_from_log(log_file):
    with open(log_file, 'r') as f:
        content = f.read()
    if "Principal axis orientation:" not in content:
        return None, []
    parts = content.split("Principal axis orientation:")
    after = parts[1]
    sections = after.split(" ---------------------------------------------------------------------")
    if len(sections) < 3:
        return None, []
    coord_block = sections[-2]
    lines = [line.strip() for line in coord_block.strip().splitlines() if line.strip()]
    if not lines:
        return None, []
    try:
        num_atoms = int(lines[-1].split()[0])
    except Exception:
        num_atoms = len(lines)
    xyz_lines = []
    for line in lines:
        tokens = line.split()
        if len(tokens) >= 5:
            atomic_num = tokens[1]
            x, y, z = tokens[2:5]
            xyz_lines.append(f"{atomic_num} {x} {y} {z}")
    return num_atoms, xyz_lines

def parse_rotational_constants_mhz(log_file):
    """
    Cerca il blocco:
      ' Rotational constants (MHZ):'
      <A> <B> <C>
    Ritorna (A, B, C) come float, oppure (None, None, None) se non trovati.
    """
    A = B = C = None
    try:
        with open(log_file, 'r', errors='ignore') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Rotational constants (MHZ):" in line:
                if i + 1 < len(lines):
                    parts = lines[i+1].strip().split()
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
    # lasciamo inalterato (identico all'originale)
    new_lines = []
    for line in lines:
        _ = re.sub(r'(?i)frozen,?', '', line)
        new_lines.append(line)
    return new_lines

def cleanup_individual_tmp(tmp_dir, individual_id):
    """Cancella i file temporanei specifici di un individuo (gjf/log/chk)."""
    base = f"individuo_{individual_id}"
    for ext in (".gjf", ".log", ".chk"):
        p = os.path.join(tmp_dir, base + ext)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

def save_statistics(file_name, generations_data):
    fields = [
        "Generation",
        "Avg. Energy", "MAX Energy", "MIN Energy",
        "Avg. HB_fitness", "MAX HB_fitness", "MIN HB_fitness",
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
        #result = subprocess.run(["gdv"], stdin=infile, stdout=outfile, stderr=subprocess.PIPE, universal_newlines=True)
        result = subprocess.run(["g16"], stdin=infile, stdout=outfile, stderr=subprocess.PIPE, universal_newlines=True)
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

def write_xyz_file(directory, individual_id, energy, hb_fitness, rank, num_atoms, xyz_lines, rotA=None, rotB=None, rotC=None):
    filename = os.path.join(directory, f"individuo_{individual_id}.xyz")
    with open(filename, 'w') as f:
        f.write(f"{num_atoms}\n")
        # aggiungiamo A/B/C se disponibili
        if rotA is not None and rotB is not None and rotC is not None:
            f.write(f"E={energy}  HB={hb_fitness}  Rank={rank}  A={rotA}  B={rotB}  C={rotC}\n")
        else:
            f.write(f"E={energy}  HB={hb_fitness}  Rank={rank}\n")
        for line in xyz_lines:
            f.write(line + "\n")
    return filename

def cleanup_tmp(directory):
    for fname in os.listdir(directory):
        if fname.endswith((".gjf", ".log", ".chk")):
            os.remove(os.path.join(directory, fname))

# ========== H-BOND DETECTION (NUOVO) ========================================

ACCEPTOR_ELEMENTS = (7, 8, 16)   # N, O, S

def identify_donors_acceptors(initial_coords, initial_bonds):
    Z = [Z for (Z,_,_,_) in initial_coords]
    donors = []
    donors_H = {}
    acceptors = [i for i,z in enumerate(Z) if z in (7,8,16)]  # N,O,S (se vuoi)
    for i, z in enumerate(Z):
        if z in (7,8,16):
            Hs = [j for j in initial_bonds[i] if Z[j] == 1]
            if Hs:
                donors.append(i)
                donors_H[i] = Hs

    # --- stampa diagnostica ---
    print("Donatori (0-based):")
    for d in donors:
        print(f"  Atomo {d+1} (Z={Z[d]}) con H legati -> {[h+1 for h in donors_H[d]]}")
    print("Accettori (0-based):", [a+1 for a in acceptors])

    return donors, donors_H, acceptors


def graph_distance_leq(bonds, i, j, max_hops: int) -> bool:
    """Ritorna True se esiste un cammino i→j di lunghezza <= max_hops nel grafo 'bonds'."""
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

def evaluate_hbond_fitness(coords, donors, donors_H, acceptors,
                           sphere=HB_SPHERE,           # non usato qui
                           bonus=HB_BONUS_PER_BOND,    # non usato qui
                           mutual_penalty=HB_MUTUAL_PENALTY,  # non usato qui
                           bonds=None):                # <<< NEW
    """
    Seconda fitness (da minimizzare):
      - Somma su tutte le interazioni (D,H,A) la 'Morse modificata' (parametri variabili).
      - AGGIUNGE una penalità H–H: per ogni H donatore vs tutti gli H del sistema,
        calcola la stessa forma funzionale con (eps=1.0, alpha=15.0, xm=2.4),
        e somma max(0, -E_HH) per penalizzare H–H troppo vicini.
    """
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

    # --- contributo D-H...A (come nella tua versione) ---
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

                # epsilon dipendente dal tipo: O–H...N > N–H...O > altri
                if Z[D] == 8 and Z[A] == 7:       # O–H...N
                    eps_eff = HB_EPS_OHN
                elif Z[D] == 7 and Z[A] == 8:     # N–H...O
                    eps_eff = HB_EPS_NHO
                elif Z[D] == 8 and Z[A] == 8:     #O-H...O
                    eps_eff = HB_EPS_OHO
                else:
                    eps_eff = HB_EPS_BASE

                Ek = hbond_pair_energy(xk, xm, alpha=15, eps=eps_eff)
                
                total_E += Ek
                if Ek < -0.3:
                    any_negative = True

                pairs_info.append({
                    "D": D+1, "H": H+1, "A": A+1,
                    "Z_D": Z[D], "Z_A": Z[A],
                    "dHA": xk, "xm": xm, "Ek": Ek
                })

    # <<< NEW: H–H penalty >>>
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

            # --- FILTRO TOPOLOGICO: escludi 1–2, 1–3, 1–4 (=> distanza <= 3 legami) ---
            if bonds is not None and graph_distance_leq(bonds, hi, hj, max_hops=4):
                continue
            xk = dist3(P[hi], P[hj])
            E_hh = 1e3
            if xk < HH_XM:
                hh_pen_sum += E_hh
            #E_hh = hbond_pair_energy(xk, HH_XM, alpha=HH_ALPHA, eps=HH_EPS)
            #hh_pen_sum += E_hh   # << sommato SEMPRE, come richiesto

            hh_pairs_info.append({
                "H1": hi+1, "H2": hj+1,
                "dHH": xk, "xm": HH_XM, "alpha": HH_ALPHA, "eps": HH_EPS,
                "E_raw": E_hh
            })

    total_E += hh_pen_sum
    # >>> END NEW

    details = {
        "pairs": pairs_info,
        "mutual": [],
        "hh_penalty": {"sum": hh_pen_sum, "pairs": hh_pairs_info}  # <<< NEW: dettagli H–H
    }
    return total_E, details, any_negative



def hbond_pair_energy(xk, xm, alpha=15.0, eps=1.0):
    """
    E_k = eps * ( exp(alpha*(1 - xk/xm))
                  - [ (xk/xm)^2 - 2*(xk/xm) + 3 ] * exp( (alpha/2)*(1 - xk/xm) ) )
    """
    if xm <= 1e-12:
        return 0.0
    t = xk / xm
    return eps * ( math.exp(alpha*(1.0 - t))
                   - (t*t - 2.0*t + 3.0) * math.exp(0.5*alpha*(1.0 - t)) )

# ========== SBX (NUOVO, angoli con wrap) ====================================

def _bounded_sbx(x1, x2, L, U, eta):
    """SBX bounded (produco UN figlio, scelgo child1/child2 a caso)."""
    if x1 > x2:
        x1, x2 = x2, x1
    if abs(x2 - x1) < 1e-12:
        return (x1 + x2) * 0.5
    u = random.random()
    # compute beta for lower bound
    beta = 1.0 + 2.0 * (x1 - L) / (x2 - x1)
    alpha = 2.0 - pow(beta, -(eta + 1.0))
    if u <= 1.0/alpha:
        betaq = pow(u * alpha, 1.0/(eta + 1.0))
    else:
        betaq = pow(1.0/(2.0 - u*alpha), 1.0/(eta + 1.0))
    child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
    # compute beta for upper bound
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

def resample_random_alleles(geni):
    """Nuovo set di alleli casuali coerenti con le periodicità dei geni."""
    return [generate_random_allele(period) for period, _ in geni]

def tweak_one_allele_random(alleles, geni, eps_deg=1e-6):
    """
    Ritorna (new_alleles, idx_mutato).
    Sceglie un gene a caso e gli assegna un nuovo allele casuale coerente
    con la periodicità. Evita (per quanto possibile) di riestrarre lo stesso valore.
    """
    if not alleles:
        return alleles, None
    i = random.randrange(len(alleles))
    period = geni[i][0]
    old = alleles[i]
    # prova qualche volta a cambiare davvero
    for _ in range(10):
        cand = generate_random_allele(period)
        if circular_diff_deg(cand, old) > eps_deg:
            break
    new = list(alleles)
    new[i] = cand
    return new, i

def sbx_crossover_angles(a1, a2, eta=SBX_ETA):
    """
    SBX su angoli (0..360) con gestione della distanza circolare:
      - si "srotola" a2 vicino ad a1 (delta in [-180, 180]),
      - si applica SBX in spazio [-180, 180] (L, U),
      - si riporta e si fa mod 360.
    """
    # delta in [-180, 180]
    delta = ((a2 - a1 + 540.0) % 360.0) - 180.0
    x1 = 0.0           # rappresento a1 come 0
    x2 = delta         # e a2 come delta
    child_rel = _bounded_sbx(x1, x2, -180.0, 180.0, eta)
    child = a1 + child_rel
    return _wrap360(child)

def crossover_sbx(parent1, parent2, eta=SBX_ETA):
    """Restituisce gli alleli del figlio usando SBX per ciascun gene angolare."""
    alleles = []
    for a1, a2 in zip(parent1["alleli"], parent2["alleli"]):
        alleles.append(sbx_crossover_angles(a1, a2, eta=eta))
    return alleles

# ========== MUTAZIONE (come prima) ===========================================

# (opzionale) cache per non ricalcolare le probabilità ad ogni chiamata
_DISCRETE_PROBS_CACHE = {}  # chiave: (periodicity, step_degrees) -> (angles_deg_list, cumprobs_list)

#TODO: Qui ti arriverà dal prof periodicità e step, ti arriverà quello che vuoi come simmetria
def generate_random_allele_discrete(periodicity: int, step_degrees: float = 10.0) -> float:
    """
    Campiona un allele angolare SOLO sui punti {0, step, 2*step, ...} in [0, 360),
    pesando ciascun punto con la stessa 'density' continua usata nella versione originale:
        density(theta) = 1 + b + (-1)^n * cos(n * theta)
    dove theta è in radianti.
    
    Ritorna SEMPRE il punto di inizio intervallo (es. 0,10,20,...,350) come float.
    """
    # Parametri come nella versione continua
    w = 0.08
    b = 1/(2*math.pi) - w
    n = periodicity

    # Normalizza lo step e costruisci la griglia discreta [0, 360)
    #TODO: Lo step si può modificare
    if step_degrees <= 0:
        step_degrees = 10.0
    # numero di punti (es. 36 per 10°). Usiamo int(round(...)) per evitare problemi di floating
    npts = max(1, int(round(360.0 / step_degrees)))
    step = 360.0 / npts  # riallinea in caso di step non divisore esatto
    angles_deg = [i * step for i in range(npts)]

    # Cache: evita di ricomputare ogni volta cumprob
    cache_key = (n, round(step, 6))
    if cache_key in _DISCRETE_PROBS_CACHE:
        angles_deg_cached, cumprobs = _DISCRETE_PROBS_CACHE[cache_key]
        # piccolo controllo: stessa lunghezza (robustezza)
        if len(angles_deg_cached) == len(angles_deg):
            r = random.random()
            # ricerca lineare (lista corta), volendo si può usare bisect
            for a, cp in zip(angles_deg_cached, cumprobs):
                if r <= cp:
                    return a
            return angles_deg_cached[-1]

    # Calcola i pesi con la stessa density, valutata AL PUNTO DI INIZIO intervallo
    weights = []
    for ang in angles_deg:
        theta = math.radians(ang)
        dens = 1.0 + b + ((-1.0) ** n) * math.cos(n * theta)
        # clamp di sicurezza per numerica: garantisci > 0
        if dens < 1e-12:
            dens = 1e-12
        weights.append(dens)

    # Normalizza a probabilità
    s = sum(weights)
    probs = [w_i / s for w_i in weights]

    # Cumulativa per campionamento
    cumprobs = []
    acc = 0.0
    for p in probs:
        acc += p
        cumprobs.append(acc)
    cumprobs[-1] = 1.0  # chiudi numericamente

    # Metti in cache
    _DISCRETE_PROBS_CACHE[cache_key] = (angles_deg, cumprobs)

    # Estrai
    r = random.random()
    for a, cp in zip(angles_deg, cumprobs):
        if r <= cp:
            return float(a)
    return float(angles_deg[-1])

def generate_random_allele(periodicity):
    return generate_random_allele_discrete(periodicity)
    w = 0.08
    b = 1/(2*math.pi) - w
    n = periodicity
    while True:
        candidate_rad = random.uniform(0, 2*math.pi)
        density = 1 + b + math.pow(-1,n)*math.cos(n * candidate_rad)
        if random.uniform(0, 1) <= density / (2+b):
            return math.degrees(candidate_rad)

#def generate_random_allele(periodicity: int, p: float = 0.5) -> float:
#    """
#    Campiona un angolo (in gradi) in modo discreto.
#    - Se periodicity == 2:
#        con probabilità p sceglie uniformemente da [0, 180]
#        con probabilità 1-p sceglie uniformemente da [90, 270]
#    - Se periodicity == 3:
#        con probabilità p sceglie uniformemente da [60, 180, 300]
#        con probabilità 1-p sceglie uniformemente da [0, 120, 240]
#    """
#    if not (0.0 <= p <= 1.0):
#        raise ValueError("p deve essere tra 0 e 1.")
#    if periodicity == 2:
#        favored = (0, 180)
#        other   = (90, 270)
#    elif periodicity == 3:
#        favored = (60, 180, 300)
#        other   = (0, 120, 240)
#    else:
#        raise ValueError("periodicity supportate: 2 o 3.")
#
#    pool = favored if random.random() < p else other
#    return float(random.choice(pool))

def mutate(alleles, mutation_rate, geni):
    new_alleles = []
    for allele, (period, _) in zip(alleles, geni):
        if random.random() < mutation_rate:
            new_alleles.append(generate_random_allele(period))
        else:
            new_alleles.append(allele)
    return new_alleles

# ========== NSGA-II (multi-obiettivo) ========================================

def dominates(a, b):
    # minimizziamo entrambe: energia e hb_fitness
    a1, a2 = a["fitness_energy"], a["fitness_hbond"]
    b1, b2 = b["fitness_energy"], b["fitness_hbond"]
    if not (math.isfinite(a1) and math.isfinite(a2) and math.isfinite(b1) and math.isfinite(b2)):
        # i non-finiti non dominano (semplice gestione)
        return False
    return (a1 <= b1 and a2 <= b2) and (a1 < b1 or a2 < b2)

def fast_non_dominated_sort(pop):
    S = {i: [] for i in range(len(pop))}
    n_dom = [0]*len(pop)
    fronts = [[]]
    for i, p in enumerate(pop):
        S[i] = []
        n_dom[i] = 0
        for j, q in enumerate(pop):
            if i == j: continue
            if dominates(p, q):
                S[i].append(j)
            elif dominates(q, p):
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
    fronts.pop()  # l'ultimo è vuoto
    return fronts

def crowding_distance(front, pop):
    if not front:
        return
    for i in front:
        pop[i]["crowding"] = 0.0
    # per ciascun obiettivo
    for obj_key in ["fitness_energy", "fitness_hbond"]:
        front_sorted = sorted(front, key=lambda idx: pop[idx][obj_key])
        # bordo
        pop[front_sorted[0]]["crowding"] = float("inf")
        pop[front_sorted[-1]]["crowding"] = float("inf")
        # range
        vmin = pop[front_sorted[0]][obj_key]
        vmax = pop[front_sorted[-1]][obj_key]
        if vmax == vmin:
            continue
        for k in range(1, len(front_sorted)-1):
            prev_val = pop[front_sorted[k-1]][obj_key]
            next_val = pop[front_sorted[k+1]][obj_key]
            pop[front_sorted[k]]["crowding"] += (next_val - prev_val) / (vmax - vmin)

def assign_pareto_metrics(pop):
    # inizializza per sicurezza
    for ind in pop:
        ind["rank"] = int(1e9)
        ind["crowding"] = 0.0
    fronts = fast_non_dominated_sort(pop)
    for f in fronts:
        crowding_distance(f, pop)
    return fronts

def selection_nsga2(population, target_size):
    """Seleziona i migliori target_size per rank crescente e crowding decrescente."""
    fronts = assign_pareto_metrics(population)
    selected = []
    for f in fronts:
        if len(selected) + len(f) <= target_size:
            selected.extend([population[i] for i in f])
        else:
            # ordina per crowding decrescente
            rest = [population[i] for i in f]
            rest.sort(key=lambda ind: ind["crowding"], reverse=True)
            selected.extend(rest[:target_size - len(selected)])
            break
    return selected

def pareto_tournament_selection(population, tournament_size=2):
    competitors = random.sample(population, tournament_size)
    # migliore: rank min, poi crowding max
    competitors.sort(key=lambda ind: (ind.get("rank", 1e9), -ind.get("crowding", 0.0)))
    return competitors[0]

# ========== POPOLAZIONE & VALUTAZIONE =======================================

def initialize_population(pop_size, geni):
    population = []
    for i in range(pop_size):
        alleli = [generate_random_allele(period) for period, _ in geni]
        population.append({
            "id": i,
            "alleli": alleli,
            # multi-obiettivo:
            "fitness_energy": None,
            "fitness_hbond": 0.0,
            "rank": None,
            "crowding": 0.0,
            # output:
            "xyz_file": None,
            "num_atoms": None,
            "rotA": None, "rotB": None, "rotC": None,
            "xyz_lines": [],
            # info aggiuntive:
            "helped": False,          # True se hb_fitness < 0
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
    """
    Valuta 'ind'. Se la topologia non coincide con quella iniziale, scarta e
    rigenera nuovi alleli per lo STESSO individuo (stesso id), riscrivendo i file,
    fino a trovare una topologia valida o a esaurire i tentativi.
    """
    # Base comune (parte invariabile del .gjf senza righe GENE)
    base_content = strip_gene_lines(INPUT_FILE)
    base_lines = remove_frozen_substring(base_content.splitlines(True))

    tries = 0
    while True:
        tries += 1

        # 1) (Ri)crea i file dell'individuo con gli ALLELI correnti
        individual_lines = add_gene_lines(base_lines, geni, ind["alleli"])
        gjf_file = write_individual_file(tmp_dir, ind["id"], individual_lines)
        log_file = os.path.join(tmp_dir, f"individuo_{ind['id']}.log")

        # 2) Esegui gdv
        retcode = run_gdv(gjf_file, log_file)
        if retcode != 0:
            # Se gdv fallisce, scarta e riprova subito con nuovi alleli
            cleanup_individual_tmp(tmp_dir, ind["id"])
            ind["alleli"], _ = tweak_some_alleles_random(ind["alleli"], geni)   # << cambia 1..n geni

            if tries >= max_topology_tries:
                # ultima spiaggia: prova a clonare da un individuo sano
                if rescue_pool is not None and clone_from_rescue(ind, rescue_pool):
                    return
                # altrimenti lascia inf come prima
                ind["fitness_energy"] = float("inf")
                ind["fitness_hbond"]  = float("inf")
                ind["num_atoms"] = None
                ind["xyz_lines"] = []
                ind["xyz_file"]  = None
                return

            continue

        # 3) Estrarre energia (la usiamo solo se la topologia risulta corretta)
        energy = parse_fitness(log_file)
        if energy is None:
            # energia non trovata => scarto e rigenero
            cleanup_individual_tmp(tmp_dir, ind["id"])
            ind["alleli"], _ = tweak_some_alleles_random(ind["alleli"], geni)   # << cambia 1..n geni

            if tries >= max_topology_tries:
                # ultima spiaggia: prova a clonare da un individuo sano
                if rescue_pool is not None and clone_from_rescue(ind, rescue_pool):
                    return
                # altrimenti lascia inf come prima
                ind["fitness_energy"] = float("inf")
                ind["fitness_hbond"]  = float("inf")
                ind["num_atoms"] = None
                ind["xyz_lines"] = []
                ind["xyz_file"]  = None
                return
            continue

        # 4) Parse coordinate e controllo topologia
        coords = parse_last_orientation_coords(log_file, use_standard=use_standard)
        if not coords:
            coords = parse_last_orientation_coords(log_file, use_standard=(not use_standard))

        if initial_bonds is not None and coords:
            current_bonds = build_bond_graph(coords)
            if (len(current_bonds) != len(initial_bonds)) or (not same_topology(current_bonds, initial_bonds)):
                # TOPOLOGIA SBAGLIATA: scarto TUTTO e rigenero nuovi alleli
                cleanup_individual_tmp(tmp_dir, ind["id"])
                ind["alleli"], _ = tweak_some_alleles_random(ind["alleli"], geni)   # << cambia 1..n geni

                if tries >= max_topology_tries:
                    # ultima spiaggia: prova a clonare da un individuo sano
                    if rescue_pool is not None and clone_from_rescue(ind, rescue_pool):
                        return
                    # altrimenti lascia inf come prima
                    ind["fitness_energy"] = float("inf")
                    ind["fitness_hbond"]  = float("inf")
                    ind["num_atoms"] = None
                    ind["xyz_lines"] = []
                    ind["xyz_file"]  = None
                    return
                continue
        else:
            # Se non ho topologia/coords, prova il clone; altrimenti invalida
            if rescue_pool is not None and clone_from_rescue(ind, rescue_pool):
                return
            ind["fitness_energy"] = float("inf")
            ind["fitness_hbond"]  = float("inf")
            ind["num_atoms"] = None
            ind["xyz_lines"] = []
            ind["xyz_file"]  = None
            return

        # 5) A questo punto la topologia è OK: assegno energia e valuto H-bond
        ind["fitness_energy"] = energy
        # Estrai le costanti rotazionali dal log
        rotA, rotB, rotC = parse_rotational_constants_mhz(log_file)
        ind["rotA"] = rotA
        ind["rotB"] = rotB
        ind["rotC"] = rotC

        hb_fit, hb_det, helped = evaluate_hbond_fitness(
            coords, donors, donors_H, acceptors,
            sphere=hb_sphere, bonus=hb_bonus, mutual_penalty=hb_mutual_penalty,
            bonds=initial_bonds   # <<< NEW: per filtrare H–H con distanza topologica
        )
        ind["fitness_hbond"] = hb_fit
        ind["hb_details"] = hb_det
        ind["helped"] = helped

        # 6) Salvo XYZ “in memoria” (lo .xyz su disco lo scrivi più avanti per i selezionati)
        num_atoms, xyz_lines = parse_xyz_from_log(log_file)
        ind["num_atoms"] = num_atoms
        ind["xyz_lines"] = xyz_lines
        ind["xyz_file"] = None

        # 7) Fine: individuo valido ottenuto
        if rescue_pool is not None and math.isfinite(ind["fitness_energy"]):
            rescue_pool.append(snapshot_for_rescue(ind))
        return



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
    parser = argparse.ArgumentParser(description="GA conformer search (multi-obiettivo: Energia + H-bond).")
    parser.add_argument("--gjf", type=str, default=INPUT_FILE, help="Percorso al file .gjf di input.")
    parser.add_argument("--out-dir", type=str, default=GENERATIONS_DIR, help="Directory output delle generazioni.")
    parser.add_argument("--tmp-dir", type=str, default=TMP_DIR, help="Directory temporanea per run intermedi.")
    parser.add_argument("--seed", type=int, default=None, help="Seed RNG (default: None).")
    parser.add_argument("--num-generazioni", type=int, default=NUM_GENERAZIONI)
    parser.add_argument("--pop-iniziale", type=int, default=POPOLAZIONE_INIZIALE)
    parser.add_argument("--pop-target", type=int, default=POPOLAZIONE_TARGET)
    parser.add_argument("--cpu-fraction", type=float, default=0.75)
    parser.add_argument("--gene-sim-threshold-deg", type=float, default=5.0)
    # (legacy, non usati)
    parser.add_argument("--near-pairs", type=str, default="")
    parser.add_argument("--near-weights", type=str, default="")
    parser.add_argument("--pair-sphere", type=float, default=PAIR_SPHERE)
    parser.add_argument("--use-standard-orientation", action="store_true")
    parser.add_argument("--use-principal-axis", action="store_true")
    parser.add_argument("--max-topology-tries", type=int, default=20,
    help="Numero massimo di tentativi di rigenerazione alleli se la topologia cambia.")
    # nuovo: H-bond objective
    parser.add_argument("--hb-sphere", type=float, default=HB_SPHERE, help="Raggio H...A per H-bond (Å).")
    parser.add_argument("--hb-bonus", type=float, default=HB_BONUS_PER_BOND, help="Quanto riduce la seconda fitness per H-bond.")
    parser.add_argument("--hb-mutual-penalty", type=float, default=HB_MUTUAL_PENALTY, help="Penalità per coppie reciproche A↔B.")
    # nuovo: SBX
    parser.add_argument("--sbx-eta", type=float, default=SBX_ETA, help="Indice di distribuzione SBX (maggiore = figli più vicini ai genitori).")
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

    # set globals
    MAX_TOPOLOGY_TRIES = int(getattr(args, "max_topology_tries", 10))
    INPUT_FILE = args.gjf
    TMP_DIR = args.tmp_dir
    GENERATIONS_DIR = args.out_dir
    NUM_GENERAZIONI = int(args.num_generazioni)
    POPOLAZIONE_INIZIALE = int(args.pop_iniziale)
    POPOLAZIONE_TARGET = int(args.pop_target)
    if args.seed is not None:
        random.seed(args.seed)

    # orientazione
    USE_STANDARD_ORIENTATION = False
    if getattr(args, "use_principal_axis", False):
        USE_STANDARD_ORIENTATION = False
    elif getattr(args, "use_standard_orientation", False):
        USE_STANDARD_ORIENTATION = True

    # parametri H-bond
    HB_SPHERE = float(getattr(args, "hb_sphere", HB_SPHERE))
    HB_BONUS_PER_BOND = float(getattr(args, "hb_bonus", HB_BONUS_PER_BOND))
    HB_MUTUAL_PENALTY = float(getattr(args, "hb_mutual_penalty", HB_MUTUAL_PENALTY))

    # SBX
    SBX_ETA = float(getattr(args, "sbx_eta", SBX_ETA))

    # (legacy NEAR_* ignorati deliberatamente)
    _ = getattr(args, "near_pairs", "")
    _ = getattr(args, "near_weights", "")
    _ = getattr(args, "pair_sphere", PAIR_SPHERE)

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(GENERATIONS_DIR, exist_ok=True)

    # log generale
    generations_data = []
    generation_log_path = os.path.join(GENERATIONS_DIR, "generation_log.txt")
    with open(generation_log_path, "w") as logf:
        logf.write("Log delle generazioni (multi-obiettivo: Energia + HB):\n\n")

    # parse GENI dal gjf (se presenti)
    parsed_geni = parse_genes_from_gjf(INPUT_FILE)
    if parsed_geni:
        GENI = parsed_geni

    reference_lines = read_reference_file(INPUT_FILE)

    # Topologia di riferimento
    _initial_coords = parse_coords_from_gjf(INPUT_FILE)
    _initial_bonds = build_bond_graph(_initial_coords) if _initial_coords else None

    # Donatori/Accettori fissati all'avvio (indici 0-based, coerenti con topologia)
    donors, donors_H, acceptors = identify_donors_acceptors(_initial_coords, _initial_bonds) if _initial_coords and _initial_bonds else ([], {}, [])

    # popolazione iniziale
    population = initialize_population(POPOLAZIONE_INIZIALE, GENI)
    
    rescue_pool = []

    for gen in range(NUM_GENERAZIONI):
        print(f"Generazione {gen}")

        # oscillazioni
        phase = (math.pi * NUM_OSCILLATIONS / (max(1, NUM_GENERAZIONI - 1))) * gen
        current_mutation_rate = BASE_RATE_MUTATION + DELTA_RATE_MUTATION * math.sin(phase)
        current_crossover_rate = BASE_RATE_CROSSOVER - DELTA_RATE_CROSSOVER * math.sin(phase)

        gen_dir = os.path.join(GENERATIONS_DIR, f"population_{gen}")
        os.makedirs(gen_dir, exist_ok=True)

        max_workers = max(1, int((os.cpu_count() or 1) * max(0.01, min(1.0, args.cpu_fraction))))
        max_workers = min(max_workers, len(population))

        # valuta in parallelo
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_individual,
                    ind, GENI, TMP_DIR, gen_dir,
                    _initial_bonds, donors, donors_H, acceptors,
                    HB_SPHERE, HB_BONUS_PER_BOND, HB_MUTUAL_PENALTY,
                    USE_STANDARD_ORIENTATION,
                    MAX_TOPOLOGY_TRIES,
                    rescue_pool                        # << nuovo argomento
                )
                for ind in population
            ]

            for future in as_completed(futures):
                future.result()

        # selezione NSGA-II (solo target_size)
        selected = selection_nsga2(population, POPOLAZIONE_TARGET)

        # statistiche SOLO sui selezionati
        E_list = [ind["fitness_energy"] for ind in selected if ind["fitness_energy"] is not None]
        HB_list = [ind["fitness_hbond"] for ind in selected if ind["fitness_hbond"] is not None]
        if E_list:
            avg_E = sum(E_list) / len(E_list)
            min_E = min(E_list); max_E = max(E_list)
        else:
            avg_E = min_E = max_E = None
        if HB_list:
            avg_HB = sum(HB_list) / len(HB_list)
            min_HB = min(HB_list); max_HB = max(HB_list)
        else:
            avg_HB = min_HB = max_HB = None

        # log generazione
        with open(generation_log_path, "a") as logf:
            logf.write(f"Generazione {gen}:\n")
            logf.write(f"  Energy   -> avg: {avg_E} | min: {min_E} | max: {max_E}\n")
            logf.write(f"  HB_fit   -> avg: {avg_HB} | min: {min_HB} | max: {max_HB}\n")
            logf.write(f"  Mutation rate: {current_mutation_rate}\n")
            logf.write(f"  Crossover rate: {current_crossover_rate}\n")
            logf.write("  Individui target (rank, crowding, E, HB):\n")
            for ind in selected:
                rotA = ind.get('rotA'); rotB = ind.get('rotB'); rotC = ind.get('rotC')
                logf.write(
                    f"    id={ind['id']} | rank={ind.get('rank')} | crowd={ind.get('crowding')} "
                    f"| E={ind['fitness_energy']} | HB={ind['fitness_hbond']}"
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
            "Avg. Energy": avg_E, "MAX Energy": max_E, "MIN Energy": min_E,
            "Avg. HB_fitness": avg_HB, "MAX HB_fitness": max_HB, "MIN HB_fitness": min_HB,
            "Mutation rate": current_mutation_rate,
            "Crossover rate": current_crossover_rate
        })

        # scrivi XYZ SOLO per selezionati
        for ind in selected:
            if ind.get("xyz_lines"):
                n_atoms = ind.get("num_atoms") or len(ind["xyz_lines"])
                ind["xyz_file"] = write_xyz_file(
                    gen_dir, ind["id"],
                    ind["fitness_energy"], ind["fitness_hbond"], ind.get("rank"),
                    n_atoms, ind["xyz_lines"],
                    rotA=ind.get("rotA"), rotB=ind.get("rotB"), rotC=ind.get("rotC")
                )
                
        # prepara nuova popolazione (genitori = selected)
        # assegna metriche anche ai selected (in caso di torneo)
        assign_pareto_metrics(selected)

        new_population = []
        while len(new_population) < POPOLAZIONE_INIZIALE:
            parent1 = pareto_tournament_selection(selected, tournament_size=2)
            parent2 = pareto_tournament_selection(selected, tournament_size=2)
            if random.random() < current_crossover_rate:
                child_alleles = crossover_sbx(parent1, parent2, eta=SBX_ETA)
            else:
                child_alleles = parent1["alleli"].copy()
            child_alleles = mutate(child_alleles, current_mutation_rate, GENI)
            new_population.append({
                "id": random.randint(1000, 9999),
                "alleli": child_alleles,
                "fitness_energy": None,
                "fitness_hbond": 0.0,
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
        # cleanup_tmp(TMP_DIR)  # opzionale

    # --- Dump cumulativi (mantenuti) ---
    cum_helped_dir = os.path.join(GENERATIONS_DIR, "cumulative_helped")
    cum_standard_dir = os.path.join(GENERATIONS_DIR, "cumulative_standard")
    os.makedirs(cum_helped_dir, exist_ok=True)
    os.makedirs(cum_standard_dir, exist_ok=True)

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
                        mE  = re.search(r"E=([-\d\.Ee+]+)", meta)
                        mHB = re.search(r"HB=([-\d\.Ee+]+)", meta)
                        mA  = re.search(r"\bA=([-\d\.Ee+]+)", meta)
                        mB  = re.search(r"\bB=([-\d\.Ee+]+)", meta)
                        mC  = re.search(r"\bC=([-\d\.Ee+]+)", meta)
                        E  = float(mE.group(1))  if mE  else float("inf")
                        HB = float(mHB.group(1)) if mHB else float("inf")
                        A  = float(mA.group(1))  if mA  else None
                        B  = float(mB.group(1))  if mB  else None
                        C  = float(mC.group(1))  if mC  else None
                    except:
                        E = float("inf"); HB = float("inf"); A = B = C = None
                    out.append((E, HB, A, B, C, path))
        return out

    all_xyz = _collect_xyz(GENERATIONS_DIR)
    helped   = [t for t in all_xyz if t[1] < -0.3]   # HB < 0
    standard = [t for t in all_xyz if not (t[1] < -0.3)]

    helped.sort(key=lambda t: (t[0], t[1]))     # usa E, poi HB
    standard.sort(key=lambda t: (t[0], t[1]))
    
    def _dump_cum(lst, out_dir, log_name):
        log_path = os.path.join(out_dir, log_name)
        with open(log_path, "w") as lf:
            lf.write(f"Elenco (ordinato per E crescente, poi HB) - n={len(lst)}\n")
            for rank, (E, HB, A, B, C, src) in enumerate(lst):
                base = os.path.basename(src)
                root, ext = os.path.splitext(base)
                dst_name = f"{root}_{rank}{ext}"
                dst_path = os.path.join(out_dir, dst_name)
                shutil.copyfile(src, dst_path)
                if A is not None and B is not None and C is not None:
                    lf.write(f"{rank:04d}  E={E}  HB={HB}  A={A}  B={B}  C={C}  file={dst_name}\n")
                else:
                    lf.write(f"{rank:04d}  E={E}  HB={HB}  file={dst_name}\n")

    _dump_cum(helped,   cum_helped_dir,   "cumulative_helped_log.txt")
    _dump_cum(standard, cum_standard_dir, "cumulative_standard_log.txt")
    
    # --- cumulative con TUTTI (helped + standard) ---  # <<< NEW
    cum_all_dir = os.path.join(GENERATIONS_DIR, "cumulative")
    os.makedirs(cum_all_dir, exist_ok=True)

    # Ordina tutto per Energia poi HB (come gli altri)
    all_sorted = sorted(all_xyz, key=lambda t: (t[0], t[1]))

    # Copia tutti e crea log completo
    _dump_cum(all_sorted, cum_all_dir, "cumulative_all_log.txt")

    energies_path = os.path.join(cum_all_dir, "cumulative_all_E.txt")
    with open(energies_path, "w") as ef:
        ef.write(f"Elenco energie (ordinato per E crescente) - n={len(all_sorted)}\n")
        for rank, (E, HB, A, B, C, src) in enumerate(all_sorted):
            base = os.path.basename(src)
            if A is not None and B is not None and C is not None:
                ef.write(f"{rank:04d}  E={E}  A={A}  B={B}  C={C}  file={base}\n")
            else:
                ef.write(f"{rank:04d}  E={E}  file={base}\n")

    with open(generation_log_path, "a") as logf:
        logf.write("=== Riepilogo finale ===\n")
        logf.write(f"  Totale selezionati HELPED (HB<0):   {len(helped)}\n")
        logf.write(f"  Totale selezionati STANDARD:        {len(standard)}\n")
        logf.write(f"  Cartella cumulative_helped:   {cum_helped_dir}\n")
        logf.write(f"  Cartella cumulative_standard: {cum_standard_dir}\n\n")
        logf.write(f"  Cartella cumulative_all:      {cum_all_dir}\n")  # <<< NEW
        logf.write("\n")


    save_statistics("evolution.csv", generations_data)
    print("Algoritmo completato.")

if __name__ == "__main__":
    genetic_algorithm()



