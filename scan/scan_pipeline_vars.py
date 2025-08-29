#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline per:
A) Parsing del "Summary of Optimized Potential Surface Scan" da un .log di Gaussian:
   - Estrae indici struttura (1..N), energia (dalla riga "Eigenvalues --"), e valori
     delle sole variabili ScanXXXX per ogni struttura.
   - Stampa la lista completa.
   - Applica filtro per ΔE (rispetto all'energia minima).
   - Deduplica strutture troppo simili sulle ScanXXXX, con soglia per componente e wrap-around.

B) Per le strutture selezionate:
   - Genera .gjf di ottimizzazione (rimuove 'Frozen,' e imposta Value= per le ScanXXXX).
   - Lancia Gaussian ("g16 file.gjf > file.log") in parallelo in modo CPU-aware.
   - Per ogni log ottimizzato:
       * verifica "Normal termination"
       * estrae l'ultima "Standard orientation" (XYZ)
       * estrae la tabella "Optimized Parameters" (tutte le GIC) e i relativi Value
       * crea un .gjf finale con geom=readallgic (senza Opt), XYZ ottimizzato e TUTTE le variabili con Value=
"""

import argparse
import os
import re
import sys
import math
import shutil
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

FLOAT_RE = re.compile(r'[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?')

# ---------- Utilità numeriche ----------

def floats_in(s: str) -> List[float]:
    """Estrae tutti i float (robusto anche se i "-" sono 'attaccati')."""
    return [float(x) for x in FLOAT_RE.findall(s)]

def circ_delta_deg(a: float, b: float) -> float:
    """Distanza circolare minima in gradi tra due angoli."""
    d = a - b
    # porta in (-180, +180]
    d = (d + 180.0) % 360.0 - 180.0
    return abs(d)

# ---------- Parsing SUMMARY dello scan ----------

def parse_scan_summary(log_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Ritorna:
      - lista di strutture: [{index: int, energy: float, scans: {ScanXXXX: value, ...}}, ...]
      - lista ordinata dei nomi delle variabili Scan presenti
    """
    out_structs: List[Dict[str, Any]] = []
    scan_names_order: List[str] = []
    in_summary = False

    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Summary of Optimized Potential Surface Scan" in line:
            in_summary = True
            i += 1
            continue

        if in_summary:
            # Cerca intestazione con indici "   1   2   3 ..."
            if re.match(r'^\s+\d+(?:\s+\d+)+\s*$', line):
                indices = [int(x) for x in line.split()]
                ncols = len(indices)

                # Cerca la riga "Eigenvalues -- ..."
                i += 1
                while i < len(lines) and "Eigenvalues --" not in lines[i]:
                    i += 1
                if i >= len(lines):
                    break
                energies = floats_in(lines[i])
                if len(energies) != ncols:
                    # Tolleriamo discrepanze solo se possiamo tagliare/estendere
                    energies = (energies + [float('nan')] * ncols)[:ncols]

                # Ora leggi le righe di parametri finché non incontri nuova intestazione,
                # riga vuota o "Largest change"/"GradGrad"
                i += 1
                block_params: Dict[str, List[float]] = {}
                while i < len(lines):
                    ln = lines[i]
                    if (re.match(r'^\s+\d+(?:\s+\d+)+\s*$', ln) or
                        "Largest change" in ln or
                        ln.strip() == "" or
                        ln.startswith(" GradGrad")):
                        break
                    m = re.match(r'^\s*([A-Za-z]{4}\d{4})\s+(.*)$', ln)
                    if m:
                        name = m.group(1)
                        vals = floats_in(m.group(2))
                        # Assicura ncols valori
                        if len(vals) < ncols:
                            # Prova a guardare la riga successiva se è "continuazione"
                            peek = lines[i+1] if i+1 < len(lines) else ""
                            cand = vals + floats_in(peek)
                            if len(cand) >= ncols:
                                vals = cand[:ncols]
                                # se abbiamo usato peek, non saltare i++
                        vals = (vals + [float('nan')] * ncols)[:ncols]
                        block_params[name] = vals
                    i += 1

                # Costruisci le strutture per questa fascia di indici
                # Ci interessano SOLO le variabili di tipo ScanXXXX
                scan_keys = [k for k in block_params.keys() if k.startswith("Scan")]
                scan_keys.sort()
                # Memorizza l'ordine dei nomi scan (una volta sola)
                if not scan_names_order:
                    scan_names_order = scan_keys

                for j, idx in enumerate(indices):
                    scans = {k: block_params[k][j] for k in scan_keys if k in block_params}
                    out_structs.append({
                        "index": idx,
                        "energy": energies[j],
                        "scans": scans
                    })
                # NON fare i += 1 qui: il while esterno avanza
                continue

            # Fine della sezione summary?
            if "GradGrad" in line:
                break

        i += 1

    # Ordina per indice crescente (1..N)
    out_structs.sort(key=lambda d: d["index"])
    return out_structs, scan_names_order

# ---------- Stampa tabelle ----------

def print_full_table(structs: List[Dict[str, Any]], scan_order: List[str]) -> None:
    header = ["Idx", "Energy"] + scan_order
    fmt = "{:>4}  {:>12.6f}  " + "  ".join(["{:>12.6f}"] * len(scan_order))
    print("\n=== TUTTE LE STRUTTURE (summary scan) ===")
    print("  ".join(f"{h:>12}" if i else f"{h:>4}" for i, h in enumerate(header)))
    for s in structs:
        row = [s["index"], s["energy"]] + [s["scans"].get(k, float('nan')) for k in scan_order]
        try:
            print(fmt.format(*row))
        except Exception:
            # In caso di NaN/None: gestiscili come 0.0 per stampa
            row2 = [row[0], float(row[1])] + [float(x) if isinstance(x, (int, float)) else float('nan') for x in row[2:]]
            print(fmt.format(*row2))

def filter_by_de(structs: List[Dict[str, Any]], de_threshold: float) -> List[Dict[str, Any]]:
    energies = [s["energy"] for s in structs]
    emin = min(energies)
    sel = []
    for s in structs:
        de = s["energy"] - emin
        if de <= de_threshold:
            t = dict(s)
            t["dE"] = de
            sel.append(t)
    sel.sort(key=lambda d: (d["dE"], d["index"]))
    return sel

def deduplicate_scans(structs: List[Dict[str, Any]],
                      scan_order: List[str],
                      angle_tol_deg: float) -> List[Dict[str, Any]]:
    """
    Greedy: ordina per energia crescente, poi scarta quelli 'troppo simili'
    (tutte le componenti entro angle_tol con wrap-around).
    """
    keep: List[Dict[str, Any]] = []
    # Ordina per dE (se presente) altrimenti energy
    structs_sorted = sorted(structs, key=lambda d: (d.get("dE", d["energy"]), d["energy"]))
    for cand in structs_sorted:
        sv = [cand["scans"].get(k, float('nan')) for k in scan_order]
        is_dup = False
        for ref in keep:
            rv = [ref["scans"].get(k, float('nan')) for k in scan_order]
            # Se tutte le componenti sono "vicine", è duplicato
            close_all = True
            for a, b in zip(sv, rv):
                if math.isnan(a) or math.isnan(b):
                    close_all = False
                    break
                if circ_delta_deg(a, b) > angle_tol_deg:
                    close_all = False
                    break
            if close_all:
                is_dup = True
                break
        if not is_dup:
            keep.append(cand)
    return keep

# ---------- Parsing/patch del GJF ----------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(path: str, txt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def parse_nprocshared(gjf_text: str, default: int = 8) -> int:
    m = re.search(r'%Nprocshared\s*=\s*(\d+)', gjf_text, flags=re.IGNORECASE)
    if not m:
        return default
    return max(1, int(m.group(1)))

# --- helper robusti per riconoscere XYZ e GIC ---
def parse_gic_atom_indices_from_gjf(original_gjf_text: str) -> Dict[str, Tuple[int, ...]]:
    """
    Estrae per ogni GIC (Stre####, Scan####, Dihe####, Bend####, ImpD####, RDef####, RPck####, SymD####, Rock####)
    gli indici atomici usati nella definizione. Esempi:
      '... = R( 1, 2)'      -> (1,2)
      '... = A( 1, 2, 14)'  -> (1,2,14)
      '... = D( 3, 1, 2,14)'-> (3,1,2,14)
      '... = [ 0.57735*D(7,8,9,10)-0.28868*D(8,9,10,13)+ ... ]' -> unione di tutti gli indici citati
    """
    _, _, gic_lines = extract_sections_from_gjf(original_gjf_text)
    name_re = re.compile(r'^\s*([A-Za-z]{4}\d{4})\b')
    # cattura numeri dentro R(...), A(...), D(...)
    idx_re = re.compile(r'[RAD]\s*\(\s*([0-9,\s]+)\s*\)', re.IGNORECASE)

    mapping: Dict[str, set] = {}
    cur_name = None
    for ln in gic_lines:
        m = name_re.match(ln)
        if m:
            cur_name = m.group(1)
            mapping.setdefault(cur_name, set())
        if cur_name:
            for block in idx_re.findall(ln):
                # es: "7, 8, 9, 10"
                try:
                    indices = tuple(int(t) for t in re.split(r'[,\s]+', block.strip()) if t)
                    mapping[cur_name].update(indices)
                except Exception:
                    pass
    # converti a tuple ordinate
    return {k: tuple(sorted(v)) for k, v in mapping.items()}

def atomic_numbers_from_gjf(gjf_text: str) -> Dict[int, int]:
    """
    Ritorna {atom_index: Z} dall'XYZ del GJF originale.
    NB: l'ordine è quello d'ingresso (1-based come nei riferimenti delle GIC).
    """
    _, xyz_lines, _ = extract_sections_from_gjf(gjf_text)
    Z_by_idx = {}
    idx = 1
    for ln in xyz_lines:
        parts = ln.split()
        if len(parts) >= 4:
            # primo token può essere simbolo o numero atomico; nel tuo GJF è il numero (6,8,1, ...)
            try:
                Z = int(parts[0])
            except ValueError:
                # se mai usassi simboli, metti un mapping simbolo->Z qui
                continue
            Z_by_idx[idx] = Z
            idx += 1
    return Z_by_idx

def is_angular_gic(name: str) -> bool:
    """Consideriamo angolari queste famiglie: Scan, Dihe, Bend, ImpD."""
    prefix = name[:4].lower()
    return prefix in ("scan", "dihe", "bend", "impd")

def gic_close(name: str, a: float, b: float, tol_angle_deg: float, tol_abs: float) -> bool:
    if is_angular_gic(name):
        return circ_delta_deg(a, b) <= tol_angle_deg
    else:
        return abs(a - b) <= tol_abs

def dedup_final_candidates(cands: List[Dict[str, Any]],
                           tol_angle_deg: float,
                           tol_abs: float,
                           ignore_names: set) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    cands_sorted = sorted(cands, key=lambda d: (d.get("dE", float("inf")), d.get("idx", 10**9)))

    def filtered_params(p: Dict[str, float]) -> Dict[str, float]:
        return {k: v for k, v in p.items() if k not in ignore_names}

    for c in cands_sorted:
        pc = filtered_params(c["params"])
        if not pc:
            # se rimuovendo ignorati non resta niente, consideralo distinto per sicurezza
            kept.append(c)
            continue
        dup = False
        for r in kept:
            pr = filtered_params(r["params"])
            if pr.keys() != pc.keys():
                continue
            # tutte le componenti devono essere vicine
            if all(gic_close(n, pc[n], pr[n], tol_angle_deg, tol_abs) for n in pc.keys()):
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept

# mapping minimo: H, C, O (estendibile se ti serve)
_ATOMIC_SYMBOL = {1: "H", 6: "C", 8: "O"}

def atomic_symbol(Z: int) -> str:
    # se non mappato, usa il numero come stringa (o aggiungi qui altri elementi)
    return _ATOMIC_SYMBOL.get(Z, str(Z))

def write_xyz_file(path: str, xyz: List[Tuple[int, float, float, float]]) -> None:
    """
    Scrive un file .xyz con:
      prima riga: numero atomi
      seconda riga: vuota
      poi righe 'Symbol  x  y  z'
    """
    n = len(xyz)
    lines = [str(n), ""]
    for Z, x, y, z in xyz:
        lines.append(f"{atomic_symbol(Z):>2s} {x:14.6f} {y:12.6f} {z:12.6f}")
    txt = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def split_header_title_charge(gjf_text: str) -> str:
    """
    Estrae SOLO l'header (directive lines + titolo + riga di carica/molteplicità).
    Esclude l'XYZ e le GIC.
    """
    lines = gjf_text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        out.append(ln)
        if re.match(r'^\s*-?\d+\s+\d+\s*$', ln):
            break  # ci fermiamo subito dopo la riga '0 1'
    return "\n".join(out) + "\n"

def is_float_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False

def looks_like_xyz_line(line: str) -> bool:
    """
    Riconosce una riga XYZ del tipo:
      - "C   -0.123   1.234   0.000"
      - "6    0.000   0.000   0.000"
    Criteri: almeno 4 token; il primo è simbolo elementare o intero; gli ultimi 3 sono float.
    """
    parts = line.strip().split()
    if len(parts) < 4:
        return False

    first = parts[0]
    # simbolo chimico (H, He, Li, ..., Og) oppure numero atomico
    is_symbol = bool(re.match(r'^[A-Z][a-z]?$', first))
    is_atomic_Z = first.isdigit() and 0 < int(first) < 200  # largo per sicurezza

    if not (is_symbol or is_atomic_Z):
        return False

    # Devono esserci almeno tre numeri dopo
    return all(is_float_token(tok) for tok in parts[1:4])

def looks_like_gic_line(line: str) -> bool:
    """
    Riconosce una riga di definizione GIC (Stre####, Scan####, Dihe####, SymD####, Rock####, RDef####, RPck####, Bend####, ImpD####, ...)
    Tipicamente: 'Name(....) = ...' o anche 'Name(....)=[ ... ]'
    """
    return bool(re.match(r'^\s*[A-Za-z]{3,6}\d{4}\s*\(', line))


def _join_xyz_and_gic_with_single_blank(xyz_lines, gic_lines) -> str:
    # rimuovi TUTTE le righe vuote dall'XYZ (anche interne)
    xyz = [ln for ln in xyz_lines if ln.strip() != ""]

    # rimuovi righe vuote di testa dal GIC
    gic = list(gic_lines)
    while gic and gic[0].strip() == "":
        gic.pop(0)

    xyz_txt = "\n".join(xyz) + "\n\n"   # termina XYZ con newline
    # tra XYZ e GIC: una sola riga vuota
    if gic:
        gic_txt = "\n".join(gic) + "\n"
    else:
        gic_txt = "\n"  # mantieni file ben terminato

    return xyz_txt + gic_txt

def patch_route_for_optimization(header: str) -> str:
    """
    Per i GJF di ottimizzazione, la route deve risultare:
      #P geom=readallgic UFF Opt=nomicro Output=Pickett

    - normalizza qualsiasi geom=... o geom=(...) -> geom=readallgic
    - rimuove eventuale 'modredundant'
    - impone Opt=nomicro (se Opt manca, lo inserisce prima di Output=Pickett se presente)
    - non tocca gli altri token (UFF, Output=Pickett, ecc.)
    """
    def _rewrite_route_line(line: str) -> str:
        # geom -> geom=readallgic
        line = re.sub(r'(?i)\bgeom\s*=\s*\([^)]*\)|\bgeom\s*=\s*\S+', 'geom=readallgic', line)
        # elimina eventuale 'modredundant' residuo
        line = re.sub(r'(?i),?\s*modredundant\b', '', line)

        # Opt -> Opt=nomicro (sostituisce qualunque forma)
        if re.search(r'(?i)\bOpt\b', line):
            line = re.sub(r'(?i)\bOpt\s*(=\s*\([^)]*\)|=\s*\S+)?', 'Opt=nomicro', line)
        else:
            # se non c'è Opt, inseriscilo prima di Output=Pickett (se presente) o in coda
            if re.search(r'(?i)\bOutput=Pickett\b', line):
                line = re.sub(r'(?i)\bOutput=Pickett\b', 'Opt=nomicro Output=Pickett', line, count=1)
            else:
                line = line.rstrip() + ' Opt=nomicro'

        # compattazione spazi
        line = re.sub(r'[ \t]+', ' ', line)
        return line

    # applica solo alla prima riga che inizia con '#'
    header2 = re.sub(r'(?m)^(#.*)$', lambda m: _rewrite_route_line(m.group(1)), header, count=1)
    return header2

# --- NUOVA versione robusta ---
def extract_sections_from_gjf(gjf_text: str) -> Tuple[str, List[str], List[str]]:
    """
    Divide il gjf in:
      - header: fino alla riga '0 1' (inclusa), e stop.
      - XYZ: dalla prima riga NON VUOTA dopo '0 1', fino a prima di una riga vuota o di una riga GIC.
      - GIC: il resto (a partire dalla prima riga che "sembra" GIC).
    """
    lines = gjf_text.splitlines()

    # 1) trova carica/molteplicità (es. "  0  1")
    idx_charge = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*-?\d+\s+\d+\s*$', ln):
            idx_charge = i
            break
    if idx_charge is None:
        raise ValueError("Impossibile individuare carica/molteplicità nel GJF (riga '0 1').")

    # 2) header = TUTTO fino a idx_charge INCLUSO (non andare oltre!)
    header = "\n".join(lines[:idx_charge + 1]) + "\n"

    # 3) inizio XYZ = prima riga NON VUOTA dopo '0 1'
    k = idx_charge + 1
    while k < len(lines) and lines[k].strip() == "":
        k += 1
    start_xyz = k

    # 4) scorri l'XYZ riga per riga
    def is_float_token(tok: str) -> bool:
        try:
            float(tok); return True
        except Exception:
            return False

    def looks_like_xyz_line(line: str) -> bool:
        parts = line.strip().split()
        if len(parts) < 4:
            return False
        first = parts[0]
        is_symbol = bool(re.match(r'^[A-Z][a-z]?$', first))
        is_atomic_Z = first.isdigit() and 0 < int(first) < 200
        if not (is_symbol or is_atomic_Z):
            return False
        return all(is_float_token(tok) for tok in parts[1:4])

    def looks_like_gic_line(line: str) -> bool:
        return bool(re.match(r'^\s*[A-Za-z]{3,6}\d{4}\s*\(', line))

    i = start_xyz
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "":
            break
        if looks_like_gic_line(ln):
            break
        if not looks_like_xyz_line(ln):
            break
        i += 1

    xyz_lines = lines[start_xyz:i]

    # 5) GIC: salta eventuali vuote, poi dal primo che sembra GIC in poi
    g = i
    while g < len(lines) and lines[g].strip() == "":
        g += 1
    if g < len(lines) and not looks_like_gic_line(lines[g]):
        # fallback: cerca la prima riga GIC dopo l'XYZ
        fallback = None
        for t in range(i, len(lines)):
            if looks_like_gic_line(lines[t]):
                fallback = t
                break
        g = fallback if fallback is not None else len(lines)

    gic_lines = lines[g:] if g is not None and g < len(lines) else []

    return header, xyz_lines, gic_lines

# --- SOSTITUISCI la vecchia strip_frozen_and_set_scan_values con questa ---

def prepare_gic_lines_for_optimization(gic_lines: List[str],
                                       scan_values: Dict[str, float]) -> List[str]:
    """
    Prepara il blocco GIC per i .gjf di ottimizzazione:
      - rimuove 'Frozen,'/'frozen,' dappertutto
      - per le righe Scan#### ricostruisce la parentesi come '(Value=<valore>)'
        eliminando qualunque altro contenuto (NSteps, StepSize, ...)
    """
    out = []
    for ln in gic_lines:
        base = ln

        # 1) rimuovi sempre 'Frozen,' (case-insensitive) nella parentesi con Value
        #    (lascia invariato tutto il resto)
        base = re.sub(r'\(\s*(?:[Ff]rozen,\s*)?Value\s*=', '(Value=', base)

        # 2) se è una riga di tipo Scan####, ricostruisci la parentesi con il Value scelto
        m = re.match(r'^(\s*)(Scan\d{4})\s*\([^)]*\)\s*(=\s*.*)$', base)
        if m:
            prefix, name, tail = m.groups()
            if name in scan_values:
                val = scan_values[name]
                # vogliamo 'Scan0001(Value=-0.049700) = D(...)'
                base = f"{prefix}{name}(Value={val:.6f}) {tail}"

        out.append(base)
    return out


def build_opt_gjf(original_gjf_text: str,
                  scan_values: Dict[str, float],
                  xyz_override: List[Tuple[int, float, float, float]] = None) -> str:
    header, xyz_lines, gic_lines = extract_sections_from_gjf(original_gjf_text)
    header = patch_route_for_optimization(header)

    # Se abbiamo un XYZ specifico per l'Idx, usalo; altrimenti usa quello del GJF originale
    if xyz_override is not None and len(xyz_override) > 0:
        xyz_clean = [f"{Z:>2d} {x:14.6f} {y:12.6f} {z:12.6f}" for Z, x, y, z in xyz_override]
    else:
        # SANITIZZA: nessuna riga vuota nell'XYZ
        xyz_clean = [ln for ln in xyz_lines if ln.strip() != ""]

    gic_patched = prepare_gic_lines_for_optimization(gic_lines, scan_values)
    # SANITIZZA: nessuna riga vuota all'inizio del GIC
    while gic_patched and gic_patched[0].strip() == "":
        gic_patched.pop(0)

    body = _join_xyz_and_gic_with_single_blank(xyz_clean, gic_patched)
    return header + body

# ---------- Lancio Gaussian in parallelo ----------

def compute_max_workers(nprocshared: int) -> int:
    try:
        ncpu = multiprocessing.cpu_count()
    except Exception:
        ncpu = 4
    # usa almeno 1, al più cpu/nprocshared
    return max(1, ncpu // max(1, nprocshared))

def run_gaussian(gexec: str, gjf_path: str, log_path: str) -> int:
    with open(log_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run([gexec, gjf_path], stdout=fout, stderr=subprocess.STDOUT)
        return proc.returncode

# ---------- Parsing log ottimizzato ----------
def parse_all_standard_orientation_xyz_blocks(text: str) -> List[List[Tuple[int, float, float, float]]]:
    """
    Estrae TUTTI i blocchi 'Standard orientation' (in ordine di apparizione)
    e ritorna una lista di blocchi, ciascuno come lista di tuple (Z, x, y, z).
    """
    import re
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if re.search(r'Standard orientation', ln, flags=re.IGNORECASE)]
    blocks: List[List[Tuple[int, float, float, float]]] = []

    for s in starts:
        i = s + 1
        # vai al primo rigo di trattini
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
            i += 1
        if i >= len(lines):
            continue
        # salta l'intestazione fino al secondo rigo di trattini
        i += 1
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
            i += 1
        if i >= len(lines):
            continue
        # prima riga dati
        i += 1

        pts: List[Tuple[int, float, float, float]] = []
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
            parts = lines[i].split()
            if len(parts) >= 6:
                try:
                    Z = int(parts[1]); x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
                    pts.append((Z, x, y, z))
                except Exception:
                    pass
            i += 1

        if pts:
            blocks.append(pts)

    return blocks



def last_standard_orientation_xyz(log_text: str) -> List[Tuple[int, float, float, float]]:
    """
    Estrae l'ULTIMO blocco 'Standard orientation' dal log Gaussian e
    ritorna una lista di tuple (Z, x, y, z).
    Struttura attesa:
        Standard orientation:
        --------------------------------------------------------------
        Center  Atomic  Atomic   X   Y   Z
        Number  Number   Type
        --------------------------------------------------------------
           1       Z    ...      x   y   z
           ...
        --------------------------------------------------------------
    """
    import re

    lines = log_text.splitlines()
    # trova l'ultima occorrenza di "Standard orientation"
    idxs = [i for i, ln in enumerate(lines) if re.search(r'Standard orientation', ln, flags=re.IGNORECASE)]
    if not idxs:
        return []

    i = idxs[-1]

    # vai alla PRIMA riga di trattini dopo il tag
    i += 1
    while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
        i += 1
    if i >= len(lines):
        return []

    # salta il blocchetto header (fino alla SECONDA riga di trattini)
    i += 1  # riga dopo i trattini
    while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
        i += 1
    if i >= len(lines):
        return []

    # la riga successiva è la PRIMA riga dati
    i += 1

    pts: List[Tuple[int, float, float, float]] = []
    while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
        parts = lines[i].split()
        if len(parts) >= 6:
            try:
                Z = int(parts[1])          # "Atomic Number"
                x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
                pts.append((Z, x, y, z))
            except Exception:
                pass
        i += 1

    return pts

def parse_optimized_parameters(log_text: str) -> Dict[str, float]:
    """
    Estrae l'ULTIMA tabella 'Optimized Parameters' in forma {Name: Value}.
    Robusta a spazi e formati; legge il numero nella colonna 'Value'.
    """
    import re
    FLOAT_RE = re.compile(r'[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?')
    lines = log_text.splitlines()
    heads = [i for i, ln in enumerate(lines) if re.search(r'Optimized Parameters', ln, re.IGNORECASE)]
    if not heads:
        raise ValueError("Nessuna tabella 'Optimized Parameters' trovata.")
    i = heads[-1]

    # vai fino alla riga con 'Name', poi salta due righe (intestazioni) e leggi finché non trovi i trattini finali
    while i < len(lines) and 'Name' not in lines[i]:
        i += 1
    i += 2

    vals: Dict[str, float] = {}
    while i < len(lines):
        ln = lines[i]
        if re.match(r'^\s*-{3,}\s*$', ln):
            break
        s = ln.strip()
        if not s.startswith('!'):
            i += 1
            continue

        content = s[1:].strip()  # togli '!' iniziale
        # taglia la parte derivativa se presente
        if '-DE/DX' in content:
            content = content.split('-DE/DX')[0]
        if content.endswith('!'):
            content = content[:-1]

        mname = re.match(r'([A-Za-z]{4}\d{4})\b', content)
        if not mname:
            i += 1
            continue
        name = mname.group(1)

        nums = FLOAT_RE.findall(content)
        if nums:
            try:
                vals[name] = float(nums[-1])
            except Exception:
                pass

        i += 1

    return vals

def has_normal_termination(log_text: str) -> bool:
    return "Normal termination" in log_text

def patch_route_for_final(header: str) -> str:
    """
    Per il GJF finale:
      - forza geom=readallgic (elimina eventuale modredundant o parentesi)
      - rimuove qualunque forma di Opt / Opt=(...) / Opt=qualcosa
      - NON tocca gli altri keyword (es. Output=Pickett, UFF, ecc.)
    """
    # 1) normalizza 'geom=' a 'geom=readallgic'
    #    (copre sia geom=(...) che geom=qualcosa)
    header2 = re.sub(r'(?i)\bgeom\s*=\s*\([^)]*\)', 'geom=readallgic', header)
    header2 = re.sub(r'(?i)\bgeom\s*=\s*(?!readallgic\b)\S+', 'geom=readallgic', header2)

    # 2) rimuovi Opt in tutte le varianti comuni
    header2 = re.sub(r'(?i)\bOpt\s*=\s*\([^)]*\)', '', header2)   # Opt=(...)
    header2 = re.sub(r'(?i)\bOpt\s*=\s*\S+', '', header2)         # Opt=qualcosa
    header2 = re.sub(r'(?i)\bOpt\b', '', header2)                 # Opt "solo"

    # compattazione spazi
    header2 = re.sub(r'[ \t]+', ' ', header2)
    header2 = re.sub(r'\s+\n', '\n', header2)
    return header2

def build_final_gjf_from_optimized(original_gjf_text: str,
                                   xyz_opt: List[Tuple[int, float, float, float]],
                                   opt_params: Dict[str, float]) -> str:
    # 1) header base dal gjf originale
    header = split_header_title_charge(original_gjf_text)
    header = patch_route_for_final(header)

    # 2) XYZ ottimizzato
    xyz_block = [f"{Z:>2d} {x:14.6f} {y:12.6f} {z:12.6f}"
                 for Z, x, y, z in xyz_opt]

    # 3) GIC dal gjf originale
    _, _, gic_lines = extract_sections_from_gjf(original_gjf_text)

    def rewrite_gic_line(ln: str) -> str:
        m = re.match(r'^(\s*)([A-Za-z]{4}\d{4})\s*\([^)]*\)\s*(=\s*.*)$', ln)
        if not m:
            m2 = re.match(r'^(\s*)([A-Za-z]{4}\d{4})\s*(=\s*.*)$', ln)
            if not m2:
                return ln
            prefix, name, tail = m2.groups()
        else:
            prefix, name, tail = m.groups()

        val = opt_params.get(name, None)
        if val is None:
            return ln  # se non ho valore aggiornato, lascio stare

        return f"{prefix}{name}(Frozen,Value={val:.6f}) {tail}"

    final_gic_lines = [rewrite_gic_line(ln) for ln in gic_lines]

    # 4) Monta il file: header + XYZ + GIC
    body = _join_xyz_and_gic_with_single_blank(xyz_block, final_gic_lines)
    return header + body

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Parsing scan + (opzionale) ottimizzazione batch con Gaussian.")
    ap.add_argument("--gjf", required=True, help="GJF originale (quello dello scan iniziale).")
    ap.add_argument("--log", required=True, help="LOG risultante dallo scan iniziale (contiene la 'Summary ...').")
    ap.add_argument("--de-threshold", type=float, default=0.050, help="Soglia ΔE = E - Emin (stesse unità della colonna Eigenvalues).")
    ap.add_argument("--angle-tol", type=float, default=5.0, help="Soglia similarità (gradi) per componente sulle ScanXXXX (wrap-around).")
    ap.add_argument("--outdir", default="OPT_JOBS", help="Cartella output per .gjf di ottimizzazione e log.")
    ap.add_argument("--gaussian-exec", default="g16", help="Eseguibile Gaussian (default: g16).")
    ap.add_argument("--run-optimization", action="store_true", help="Esegue anche la fase B (ottimizzazione + gjf finali).")
    ap.add_argument("--final-angle-tol", type=float, default=2.0,
                help="Tolleranza (gradi) per confronto GIC angolari nella dedup finale.")
    ap.add_argument("--final-abs-tol", type=float, default=0.01,
                help="Tolleranza assoluta per GIC non angolari nella dedup finale.")
    args = ap.parse_args()

    # --- A) Parsing summary ---
    gjf_text = read_text(args.gjf)
    log_text = read_text(args.log)
    structs, scan_order = parse_scan_summary(log_text)

    if not structs:
        print("ERRORE: nessuna struttura trovata nella Summary dello scan.", file=sys.stderr)
        sys.exit(1)
    if not scan_order:
        print("ATTENZIONE: nessuna variabile ScanXXXX trovata nella Summary.", file=sys.stderr)

    print_full_table(structs, scan_order)

    # filtro per ΔE
    filtered = filter_by_de(structs, args.de_threshold)
    # dedup per scans
    survivors = deduplicate_scans(filtered, scan_order, args.angle_tol)

    # stampa selezionati
    print("\n=== SELEZIONATI dopo filtro ΔE e dedup scans ===")
    hdr = ["Idx", "dE"] + scan_order
    print("  ".join(f"{h:>12}" if i else f"{h:>4}" for i, h in enumerate(hdr)))
    fmt = "{:>4}  {:>12.6f}  " + "  ".join(["{:>12.6f}"] * len(scan_order))
    for s in survivors:
        row = [s["index"], s["dE"]] + [s["scans"].get(k, float('nan')) for k in scan_order]
        print(fmt.format(*row))

    if not args.run_optimization:
        return

    # --- B) Ottimizzazione in batch ---
    nprocshared = parse_nprocshared(gjf_text, default=8)
    max_workers = compute_max_workers(nprocshared)
    os.makedirs(args.outdir, exist_ok=True)

    # >>> NUOVO: estrai gli XYZ per ogni idx dal LOG dello scan, non dal GJF
    all_xyz_blocks = parse_all_standard_orientation_xyz_blocks(log_text)
    N = len(structs)  # strutture nel summary
    idx_sorted = [s["index"] for s in structs]  # di solito 1..N

    if len(all_xyz_blocks) >= N:
        # prendi gli ULTIMI N blocchi (il log spesso contiene un orientamento iniziale extra)
        #blocks_for_summary = all_xyz_blocks[-N:]
        blocks_for_summary = all_xyz_blocks[:N]
        # mappa idx -> blocco con stesso ordine dell'elenco 'structs' (ordinato per index)
        idx_to_xyz = {idx_sorted[i]: blocks_for_summary[i] for i in range(N)}
        print(f"[INFO] Trovati {len(all_xyz_blocks)} blocchi 'Standard orientation' nel LOG; uso gli ultimi {N} (Idx 1..{N}).")
    else:
        idx_to_xyz = {}
        print(f"[WARN] Solo {len(all_xyz_blocks)} blocchi 'Standard orientation' nel LOG, ma il summary ha {N} strutture. Userò l'XYZ del GJF come fallback.")

    print(f"[INFO] Genero e lancio {len(survivors)} job in {args.outdir}/")

    jobs = []
    for s in survivors:
       idx = s["index"]
       scan_values = s["scans"]
       xyz_override = idx_to_xyz.get(idx)  # potrebbe essere None se non presente

       gjf_opt_text = build_opt_gjf(gjf_text, scan_values, xyz_override=xyz_override)
       gjf_name = f"opt_idx{idx:03d}.gjf"
       log_name = f"opt_idx{idx:03d}.log"
       gjf_path = os.path.join(args.outdir, gjf_name)
       log_path = os.path.join(args.outdir, log_name)
       write_text(gjf_path, gjf_opt_text)
       jobs.append((gjf_path, log_path))

    # 2) lancia Gaussian in parallelo (CPU-aware)
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2job = {ex.submit(run_gaussian, args.gaussian_exec, gjf, log): (gjf, log) for gjf, log in jobs}
        for fut in as_completed(fut2job):
            gjf_path, log_path = fut2job[fut]
            rc = fut.result()
            results[gjf_path] = (log_path, rc)
            print(f"[DONE] {os.path.basename(gjf_path)} -> rc={rc}")

    # 3) per ogni log: verifica, estrae XYZ + Optimized Parameters e prepara candidati finali
    final_dir = os.path.join(args.outdir, "FINAL_GJF")
    os.makedirs(final_dir, exist_ok=True)

    final_xyz_dir = os.path.join(args.outdir, "FINAL_XYZ")
    os.makedirs(final_xyz_dir, exist_ok=True)

    # mappa idx -> dE dai selezionati, per preferire strutture più basse in energia
    de_by_idx = {s["index"]: s.get("dE", float("inf")) for s in survivors}

    # raccogli tutti i candidati completi prima della dedup
    candidates = []
    for gjf_path, (log_path, rc) in results.items():
        base = os.path.splitext(os.path.basename(gjf_path))[0]  # es. opt_idx005
        log_txt = read_text(log_path)
        if not has_normal_termination(log_txt):
            print(f"[WARN] {base}: ottimizzazione NON terminata normalmente, salto.")
            continue
        try:
            xyz_opt = last_standard_orientation_xyz(log_txt)
            params  = parse_optimized_parameters(log_txt)
            # estrai idx dal nome file
            m = re.search(r'idx(\d+)', base)
            idx = int(m.group(1)) if m else 10**9
            candidates.append({
                "idx": idx,
                "name": base,
                "xyz": xyz_opt,
                "params": params,
                "dE": de_by_idx.get(idx, float("inf"))
            })
        except Exception as e:
            print(f"[ERR] {base}: parsing log ottimizzato fallito: {e}")

    if not candidates:
        print("[INFO] Nessun candidato finale disponibile.")
        return

    # dedup finale sui parametri ottimizzati
    Z_by_idx = atomic_numbers_from_gjf(gjf_text)
    gic_atom_map = parse_gic_atom_indices_from_gjf(gjf_text)

    ignore_gic = set()
    for name, idxs in gic_atom_map.items():
        if any(Z_by_idx.get(i) == 1 for i in idxs):  # se coinvolge qualunque H
            ignore_gic.add(name)

    print(f"[INFO] Ignoro {len(ignore_gic)} GIC che coinvolgono H in dedup finale.")
    
    kept = dedup_final_candidates(candidates, args.final_angle_tol, args.final_abs_tol, ignore_gic)
    print(f"[INFO] Finali: {len(candidates)} candidati -> {len(kept)} unici (tol_angle={args.final_angle_tol}°, tol_abs={args.final_abs_tol}).")

    # scrivi solo i NON duplicati
    for c in kept:
        # GJF finale
        final_gjf = build_final_gjf_from_optimized(gjf_text, c["xyz"], c["params"])
        out_gjf = os.path.join(final_dir, c["name"].replace("opt_", "final_") + ".gjf")
        write_text(out_gjf, final_gjf)

        # XYZ finale (solo coordinate)
        out_xyz = os.path.join(final_xyz_dir, c["name"].replace("opt_", "final_") + ".xyz")
        write_xyz_file(out_xyz, c["xyz"])

        print(f"[OK]  {c['name']}: creati\n       {out_gjf}\n       {out_xyz}")

if __name__ == "__main__":
    main()


