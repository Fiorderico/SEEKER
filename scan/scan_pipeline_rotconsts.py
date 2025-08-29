#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline scan → preselezione → ottimizzazione → finali

DEDUP SU COSTANTI ROTAZIONALI (MHz)
- criterio: |Ai - Aj|/max(Ai, Aj) <= pct, e idem per B e C (default pct=0.05%)
- applicato nella preselezione (dopo ΔE) e nella dedup finale (post-ottimizzazione)
- se duplicati: si tiene SEMPRE quello a energia più bassa
  * preselezione: usa ΔE dal summary dello scan
  * finale: usa energia dall'ottimizzazione (ultimo 'SCF Done: E(...) = ...', fallback al valore dopo 'HF=')

XYZ PER L'OTTIMIZZAZIONE
- per ogni Idx si usa l'XYZ del LOG dello scan (blocco 'Standard orientation')
- scelta blocchi: --pick-blocks first|last (default: first)

ROUTE
- ottimizzazione: '#P geom=readallgic UFF Opt=nomicro Output=Pickett'
- finali: geom=readallgic, senza Opt; GIC con 'Frozen,Value=' dai log ottimizzati
"""

import argparse
import os
import re
import sys
import math
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

# -------- Regex utili ----------
FLOAT_RE = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?')

def floats_in(s: str) -> List[float]:
    return [float(x.replace('D','E').replace('d','e')) for x in FLOAT_RE.findall(s)]

def circ_delta_deg(a: float, b: float) -> float:
    d = a - b
    d = (d + 180.0) % 360.0 - 180.0
    return abs(d)

# ---------- Parsing SUMMARY dello scan ----------

def parse_scan_summary(log_text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
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
            if re.match(r'^\s+\d+(?:\s+\d+)+\s*$', line):
                indices = [int(x) for x in line.split()]
                ncols = len(indices)
                # energia
                i += 1
                while i < len(lines) and "Eigenvalues --" not in lines[i]:
                    i += 1
                if i >= len(lines): break
                energies = floats_in(lines[i])
                energies = (energies + [float('nan')]*ncols)[:ncols]
                # parametri blocco
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
                        if len(vals) < ncols:
                            peek = lines[i+1] if i+1 < len(lines) else ""
                            cand = vals + floats_in(peek)
                            if len(cand) >= ncols:
                                vals = cand[:ncols]
                        vals = (vals + [float('nan')]*ncols)[:ncols]
                        block_params[name] = vals
                    i += 1

                scan_keys = [k for k in block_params.keys() if k.startswith("Scan")]
                scan_keys.sort()
                if not scan_names_order:
                    scan_names_order = scan_keys

                for j, idx in enumerate(indices):
                    scans = {k: block_params[k][j] for k in scan_keys if k in block_params}
                    out_structs.append({"index": idx, "energy": energies[j], "scans": scans})
                continue

            if "GradGrad" in line:
                break
        i += 1

    out_structs.sort(key=lambda d: d["index"])
    return out_structs, scan_names_order

# ---------- Rotational constants (A,B,C) parsing ----------

# Etichetta (senza aspettarsi i numeri sulla stessa riga)
ROTCONST_LABEL_RE = re.compile(
    r'Rotational constants\s*\(\s*(GHZ|MHZ)\s*\)\s*:',
    re.IGNORECASE
)

def parse_all_rotconst_blocks(text: str) -> List[Tuple[float,float,float]]:
    """
    Ritorna tutte le terne (A,B,C) in MHz nell'ordine di apparizione.
    Gestisce sia il caso 'label + numeri sulla stessa riga' sia 'label su una riga,
    numeri sulla riga successiva' (eventualmente guardiamo anche la seconda riga successiva).
    """
    lines = text.splitlines()
    out: List[Tuple[float,float,float]] = []
    for i, ln in enumerate(lines):
        m = ROTCONST_LABEL_RE.search(ln)
        if not m:
            continue
        unit = m.group(1).upper()  # 'GHZ' o 'MHZ'
        # Prova a leggere numeri sulla stessa riga
        nums = floats_in(ln)
        # Se non bastano, prova sulla riga seguente (e, se serve, su quella dopo ancora)
        if len(nums) < 3 and i+1 < len(lines):
            nums += floats_in(lines[i+1])
        if len(nums) < 3 and i+2 < len(lines):
            nums += floats_in(lines[i+2])

        if len(nums) >= 3:
            a, b, c = nums[:3]
            if unit == 'GHZ':
                a, b, c = a*1000.0, b*1000.0, c*1000.0  # converti in MHz
            out.append((a, b, c))
    return out

def parse_last_rotconst_mhz(text: str) -> Tuple[float,float,float]:
    """Comodità: l'ULTIMA terna (A,B,C) in MHz (o (nan, nan, nan) se assente)."""
    vals = parse_all_rotconst_blocks(text)
    return vals[-1] if vals else (float('nan'),)*3

def rotc_close_3(a1: float, b1: float, c1: float,
                 a2: float, b2: float, c2: float,
                 pct: float) -> bool:
    """|x1-x2|/max(x1,x2) <= pct/100 per x in {A,B,C}."""
    def ok(u, v):
        m = max(abs(u), abs(v))
        if m == 0:
            return abs(u - v) == 0.0
        return abs(u - v)/m <= (pct/100.0)
    return ok(a1,a2) and ok(b1,b2) and ok(c1,c2)

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
            row2 = [row[0], float(row[1])] + [float(x) if isinstance(x,(int,float)) else float('nan') for x in row[2:]]
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

def dedup_by_rotconst(structs: List[Dict[str, Any]],
                      idx_to_rotc: Dict[int, Tuple[float,float,float]],
                      pct: float) -> List[Dict[str, Any]]:
    """
    Deduplica greedy sui rotazionali:
      - ordina per dE crescente, poi index (così si tiene la più bassa energia)
      - scarta successivi che hanno (A,B,C) entro pct% rispetto a uno già tenuto
    """
    kept: List[Dict[str, Any]] = []
    sorted_structs = sorted(structs, key=lambda d: (d.get("dE", float('inf')), d["index"]))
    for s in sorted_structs:
        idx = s["index"]
        rotc = idx_to_rotc.get(idx, None)
        if rotc is None:
            kept.append(s)  # se mancano, non rischiamo falsi scarti
            continue
        a1,b1,c1 = rotc
        dup = False
        for r in kept:
            r_rotc = idx_to_rotc.get(r["index"], None)
            if r_rotc is None: 
                continue
            a2,b2,c2 = r_rotc
            if rotc_close_3(a1,b1,c1,a2,b2,c2,pct):
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept

# ---------- GJF helpers ----------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(path: str, txt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def parse_nprocshared(gjf_text: str, default: int = 8) -> int:
    m = re.search(r'%Nprocshared\s*=\s*(\d+)', gjf_text, flags=re.IGNORECASE)
    if not m: return default
    return max(1, int(m.group(1)))

def is_float_token(tok: str) -> bool:
    try: float(tok); return True
    except Exception: return False

def looks_like_xyz_line(line: str) -> bool:
    parts = line.strip().split()
    if len(parts) < 4: return False
    first = parts[0]
    is_symbol = bool(re.match(r'^[A-Z][a-z]?$', first))
    is_atomic_Z = first.isdigit() and 0 < int(first) < 200
    if not (is_symbol or is_atomic_Z): return False
    return all(is_float_token(tok) for tok in parts[1:4])

def looks_like_gic_line(line: str) -> bool:
    return bool(re.match(r'^\s*[A-Za-z]{3,6}\d{4}\s*\(', line))

def _join_xyz_and_gic_with_single_blank(xyz_lines, gic_lines) -> str:
    xyz = [ln for ln in xyz_lines if ln.strip() != ""]
    gic = list(gic_lines)
    while gic and gic[0].strip() == "": gic.pop(0)
    xyz_txt = "\n".join(xyz) + "\n\n"
    gic_txt = ("\n".join(gic) + "\n") if gic else "\n"
    return xyz_txt + gic_txt

def patch_route_for_optimization(header: str) -> str:
    """#P geom=readallgic UFF Opt=nomicro Output=Pickett"""
    def _rewrite(line: str) -> str:
        line = re.sub(r'(?i)\bgeom\s*=\s*\([^)]*\)|\bgeom\s*=\s*\S+', 'geom=readallgic', line)
        line = re.sub(r'(?i),?\s*modredundant\b', '', line)
        if re.search(r'(?i)\bOpt\b', line):
            line = re.sub(r'(?i)\bOpt\s*(=\s*\([^)]*\)|=\s*\S+)?', 'Opt=nomicro', line)
        else:
            if re.search(r'(?i)\bOutput=Pickett\b', line):
                line = re.sub(r'(?i)\bOutput=Pickett\b', 'Opt=nomicro Output=Pickett', line, count=1)
            else:
                line = line.rstrip() + ' Opt=nomicro'
        return re.sub(r'[ \t]+', ' ', line)
    return re.sub(r'(?m)^(#.*)$', lambda m: _rewrite(m.group(1)), header, count=1)

def extract_sections_from_gjf(gjf_text: str) -> Tuple[str, List[str], List[str]]:
    lines = gjf_text.splitlines()
    idx_charge = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*-?\d+\s+\d+\s*$', ln):
            idx_charge = i; break
    if idx_charge is None:
        raise ValueError("Carica/molteplicità ('0 1') non trovata.")
    header = "\n".join(lines[:idx_charge + 1]) + "\n"
    k = idx_charge + 1
    while k < len(lines) and lines[k].strip() == "":
        k += 1
    start_xyz = k
    i = start_xyz
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "": break
        if looks_like_gic_line(ln): break
        if not looks_like_xyz_line(ln): break
        i += 1
    xyz_lines = lines[start_xyz:i]
    g = i
    while g < len(lines) and lines[g].strip() == "":
        g += 1
    if g < len(lines) and not looks_like_gic_line(lines[g]):
        fallback = None
        for t in range(i, len(lines)):
            if looks_like_gic_line(lines[t]):
                fallback = t; break
        g = fallback if fallback is not None else len(lines)
    gic_lines = lines[g:] if g is not None and g < len(lines) else []
    return header, xyz_lines, gic_lines

def prepare_gic_lines_for_optimization(gic_lines: List[str],
                                       scan_values: Dict[str, float]) -> List[str]:
    out = []
    for ln in gic_lines:
        base = re.sub(r'\(\s*(?:[Ff]rozen,\s*)?Value\s*=', '(Value=', ln)
        m = re.match(r'^(\s*)(Scan\d{4})\s*\([^)]*\)\s*(=\s*.*)$', base)
        if m:
            prefix, name, tail = m.groups()
            if name in scan_values:
                val = scan_values[name]
                base = f"{prefix}{name}(Value={val:.6f}) {tail}"
        out.append(base)
    return out

def build_opt_gjf(original_gjf_text: str,
                  scan_values: Dict[str, float],
                  xyz_override: List[Tuple[int, float, float, float]] = None) -> str:
    header, xyz_lines, gic_lines = extract_sections_from_gjf(original_gjf_text)
    header = patch_route_for_optimization(header)
    if xyz_override:
        xyz_clean = [f"{Z:>2d} {x:14.6f} {y:12.6f} {z:12.6f}" for Z,x,y,z in xyz_override]
    else:
        xyz_clean = [ln for ln in xyz_lines if ln.strip() != ""]
    gic_patched = prepare_gic_lines_for_optimization(gic_lines, scan_values)
    while gic_patched and gic_patched[0].strip() == "": gic_patched.pop(0)
    body = _join_xyz_and_gic_with_single_blank(xyz_clean, gic_patched)
    return header + body

def split_header_title_charge(gjf_text: str) -> str:
    lines = gjf_text.splitlines()
    out = []
    for ln in lines:
        out.append(ln)
        if re.match(r'^\s*-?\d+\s+\d+\s*$', ln):
            break
    return "\n".join(out) + "\n"

def patch_route_for_final(header: str) -> str:
    header2 = re.sub(r'(?i)\bgeom\s*=\s*\([^)]*\)', 'geom=readallgic', header)
    header2 = re.sub(r'(?i)\bgeom\s*=\s*(?!readallgic\b)\S+', 'geom=readallgic', header2)
    header2 = re.sub(r'(?i)\bOpt\s*=\s*\([^)]*\)', '', header2)
    header2 = re.sub(r'(?i)\bOpt\s*=\s*\S+', '', header2)
    header2 = re.sub(r'(?i)\bOpt\b', '', header2)
    header2 = re.sub(r'[ \t]+', ' ', header2)
    header2 = re.sub(r'\s+\n', '\n', header2)
    return header2

def build_final_gjf_from_optimized(original_gjf_text: str,
                                   xyz_opt: List[Tuple[int, float, float, float]],
                                   opt_params: Dict[str, float]) -> str:
    header = split_header_title_charge(original_gjf_text)
    header = patch_route_for_final(header)
    xyz_block = [f"{Z:>2d} {x:14.6f} {y:12.6f} {z:12.6f}" for Z,x,y,z in xyz_opt]

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
            mval = re.search(r'\(\s*[^)]*Value\s*=\s*([-\d\.EeDd]+)', ln, flags=re.IGNORECASE)
            if mval:
                try: val = float(mval.group(1).replace('D','E'))
                except: val = None
        if val is None or (isinstance(val,float) and math.isnan(val)):
            return ln
        return f"{prefix}{name}(Frozen,Value={val:.6f}) {tail}"

    final_gic_lines = [rewrite_gic_line(ln) for ln in gic_lines]
    while final_gic_lines and final_gic_lines[0].strip() == "": final_gic_lines.pop(0)
    body = _join_xyz_and_gic_with_single_blank(xyz_block, final_gic_lines)
    return header + body

# ---------- Parsing LOG ottimizzato ----------

def last_standard_orientation_xyz(log_text: str) -> List[Tuple[int, float, float, float]]:
    lines = log_text.splitlines()
    idxs = [i for i, ln in enumerate(lines) if re.search(r'Standard orientation', ln, flags=re.IGNORECASE)]
    if not idxs:
        raise ValueError("Nessuna 'Standard orientation' trovata nel log.")
    i = idxs[-1]
    i += 1
    while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]): i += 1
    if i >= len(lines): return []
    i += 1
    while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]): i += 1
    if i >= len(lines): return []
    i += 1
    pts: List[Tuple[int,float,float,float]] = []
    while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
        parts = lines[i].split()
        if len(parts) >= 6:
            try:
                Z = int(parts[1]); x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
                pts.append((Z,x,y,z))
            except Exception:
                pass
        i += 1
    return pts

def parse_optimized_parameters(log_text: str) -> Dict[str, float]:
    lines = log_text.splitlines()
    heads = [i for i, ln in enumerate(lines) if re.search(r'Optimized Parameters', ln, flags=re.IGNORECASE)]
    if not heads:
        raise ValueError("Nessuna tabella 'Optimized Parameters' trovata.")
    i0 = heads[-1]
    i = i0
    while i < len(lines) and 'Name' not in lines[i]: i += 1
    i += 2
    vals: Dict[str,float] = {}
    while i < len(lines):
        ln = lines[i]
        if re.match(r'^\s*-{3,}\s*$', ln): break
        s = ln.strip()
        if not s.startswith('!'):
            i += 1; continue
        content = s[1:].strip()
        if '-DE/DX' in content: content = content.split('-DE/DX')[0]
        if content.endswith('!'): content = content[:-1]
        mname = re.match(r'([A-Za-z]{4}\d{4})\b', content)
        if mname:
            name = mname.group(1)
            nums = FLOAT_RE.findall(content)
            if nums:
                val_str = nums[-1].replace('D','E')
                try: vals[name] = float(val_str)
                except: pass
        i += 1
    return vals

def has_normal_termination(log_text: str) -> bool:
    return "Normal termination" in log_text

# --- energia ottimizzata dal log (prima 'SCF Done', fallback 'HF=') ---
SCF_DONE_RE = re.compile(r'SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)', re.IGNORECASE)
HF_RE       = re.compile(r'HF=\s*([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)')

def extract_last_energy_eh(log_text: str):
    scf_all = [float(m.group(1)) for m in SCF_DONE_RE.finditer(log_text)]
    if scf_all:
        return scf_all[-1]
    hf_all = [float(m.group(1)) for m in HF_RE.finditer(log_text)]
    if hf_all:
        return hf_all[-1]
    return None

# ---------- LOG scan: orientazioni e rotazionali ----------

def parse_all_standard_orientation_xyz_blocks(text: str) -> List[List[Tuple[int, float, float, float]]]:
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if re.search(r'Standard orientation', ln, flags=re.IGNORECASE)]
    blocks: List[List[Tuple[int, float, float, float]]] = []
    for s in starts:
        i = s + 1
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]): i += 1
        if i >= len(lines): continue
        i += 1
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]): i += 1
        if i >= len(lines): continue
        i += 1
        pts: List[Tuple[int,float,float,float]] = []
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
            parts = lines[i].split()
            if len(parts) >= 6:
                try:
                    Z = int(parts[1]); x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
                    pts.append((Z,x,y,z))
                except Exception:
                    pass
            i += 1
        if pts: blocks.append(pts)
    return blocks

# ---------- XYZ writer ----------

_ATOMIC_SYMBOL = {1: "H", 6: "C", 8: "O"}  # estendi se serve

def atomic_symbol(Z: int) -> str:
    return _ATOMIC_SYMBOL.get(Z, str(Z))

def write_xyz_file(path: str, xyz: List[Tuple[int, float, float, float]]) -> None:
    n = len(xyz)
    lines = [str(n), ""]
    for Z, x, y, z in xyz:
        lines.append(f"{atomic_symbol(Z):>2s} {x:14.6f} {y:12.6f} {z:12.6f}")
    txt = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

# ---------- CPU / Gaussian runner ----------

def compute_max_workers(nprocshared: int) -> int:
    try: ncpu = multiprocessing.cpu_count()
    except Exception: ncpu = 4
    return max(1, ncpu // max(1, nprocshared))

def run_gaussian(gexec: str, gjf_path: str, log_path: str) -> int:
    with open(log_path, "w", encoding="utf-8") as fout:
        proc = subprocess.run([gexec, gjf_path], stdout=fout, stderr=subprocess.STDOUT)
        return proc.returncode

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Parsing scan + ottimizzazione batch con dedup su costanti rotazionali.")
    ap.add_argument("--gjf", required=True, help="GJF originale (quello dello scan iniziale).")
    ap.add_argument("--log", required=True, help="LOG risultato dello scan iniziale (contiene Summary + blocchi Standard orientation).")
    ap.add_argument("--de-threshold", type=float, default=0.050, help="Soglia ΔE = E - Emin.")
    ap.add_argument("--angle-tol", type=float, default=5.0, help="(legacy, non usata nella dedup corrente).")
    ap.add_argument("--outdir", default="OPT_JOBS", help="Cartella output per .gjf di ottimizzazione e log.")
    ap.add_argument("--gaussian-exec", default="g16", help="Eseguibile Gaussian (default: g16).")
    ap.add_argument("--run-optimization", action="store_true", help="Esegue anche la fase B (ottimizzazione + finali).")

    # blocchi da usare (per mappare Idx→XYZ e Idx→rotazionali) dai blocchi del LOG dello scan
    ap.add_argument("--pick-blocks", choices=["first","last"], default="first",
                    help="Quali blocchi 'Standard orientation' del LOG usare per Idx: 'first' o 'last'. Default: first.")

    # soglia percentuale per dedup su rotazionali (MHz)
    ap.add_argument("--rotc-threshold-pct", type=float, default=0.05,
                    help="Soglia percentuale per ciascuna costante rotazionale (A,B,C) in MHz (default 0.05%%).")

    args = ap.parse_args()

    # --- A) Parsing summary ---
    gjf_text = read_text(args.gjf)
    log_text = read_text(args.log)
    structs, scan_order = parse_scan_summary(log_text)
    if not structs:
        print("ERRORE: nessuna struttura trovata nella Summary dello scan.", file=sys.stderr)
        sys.exit(1)

    print_full_table(structs, scan_order)

    # mappatura Idx->XYZ e Idx->Rotazionali dai blocchi del LOG
    all_xyz_blocks = parse_all_standard_orientation_xyz_blocks(log_text)
    all_rotconst    = parse_all_rotconst_blocks(log_text)
    N = len(structs)
    if len(all_xyz_blocks) >= N:
        blocks_xyz = all_xyz_blocks[:N] if args.pick_blocks == "first" else all_xyz_blocks[-N:]
    else:
        blocks_xyz = []
    if len(all_rotconst) >= N:
        blocks_rot = all_rotconst[:N] if args.pick_blocks == "first" else all_rotconst[-N:]
    else:
        blocks_rot = []

    idx_sorted = [s["index"] for s in structs]  # tipicamente 1..N ordinati
    idx_to_xyz = {idx_sorted[i]: blocks_xyz[i] for i in range(min(len(blocks_xyz), N))}
    idx_to_rot = {idx_sorted[i]: blocks_rot[i] for i in range(min(len(blocks_rot), N))}

    print(f"[INFO] Blocchi Standard orientation nel LOG: {len(all_xyz_blocks)}; uso {len(idx_to_xyz)} per Idx.")
    print(f"[INFO] Blocchi Rotational constants nel LOG: {len(all_rotconst)}; uso {len(idx_to_rot)} per Idx.")

    # Filtro per ΔE
    filtered = filter_by_de(structs, args.de_threshold)

    # Dedup SU ROTAZIONALI (MHz) nella preselezione (tiene la più bassa ΔE)
    survivors = dedup_by_rotconst(filtered, idx_to_rot, args.rotc_threshold_pct)

    # Stampa selezionati + rotazionali (se disponibili)
    print("\n=== SELEZIONATI dopo filtro ΔE e dedup rotazionali ===")
    print(f"{'Idx':>4}  {'dE':>12}  {'A(MHz)':>12}  {'B(MHz)':>12}  {'C(MHz)':>12}")
    for s in survivors:
        idx = s["index"]
        rot = idx_to_rot.get(idx, (float('nan'),)*3)
        print(f"{idx:>4}  {s['dE']:12.6f}  {rot[0]:12.6f}  {rot[1]:12.6f}  {rot[2]:12.6f}")

    if not args.run_optimization:
        return

    # --- B) Ottimizzazione in batch ---
    nprocshared = parse_nprocshared(gjf_text, default=8)
    max_workers = compute_max_workers(nprocshared)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n[INFO] Nprocshared={nprocshared} -> max_workers={max_workers}")
    print(f"[INFO] Genero e lancio {len(survivors)} job in {args.outdir}/")

    # 1) genera i .gjf per ottimizzazione (XYZ specifico per idx se disponibile)
    jobs = []
    for s in survivors:
        idx = s["index"]
        scan_values = s["scans"]
        xyz_override = idx_to_xyz.get(idx)
        # opzionale: sanity-check numero atomi tra override e GJF base
        if xyz_override:
            try:
                _, xyz_lines_ref, _ = extract_sections_from_gjf(gjf_text)
                n_ref = sum(1 for ln in xyz_lines_ref if ln.strip())
                if len(xyz_override) != n_ref:
                    print(f"[WARN] idx={idx}: num atomi mismatch (LOG {len(xyz_override)} vs GJF {n_ref}). Fallback XYZ del GJF.")
                    xyz_override = None
            except Exception:
                pass

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

    # 3) finalizzazione: parse log, dedup finale SU ROTAZIONALI (tieni la più bassa energia), salva .gjf e .xyz
    final_dir = os.path.join(args.outdir, "FINAL_GJF")
    final_xyz_dir = os.path.join(args.outdir, "FINAL_XYZ")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(final_xyz_dir, exist_ok=True)

    # mappa idx->dE originale per eventuale fallback
    de_by_idx = {s["index"]: s.get("dE", float("inf")) for s in survivors}

    # raccogli candidati
    candidates = []
    for gjf_path, (log_path, rc) in results.items():
        base = os.path.splitext(os.path.basename(gjf_path))[0]  # opt_idxNNN
        log_txt = read_text(log_path)
        if not has_normal_termination(log_txt):
            print(f"[WARN] {base}: ottimizzazione NON terminata normalmente, salto.")
            continue
        try:
            xyz_opt = last_standard_orientation_xyz(log_txt)
            params  = parse_optimized_parameters(log_txt)
            # rotazionali dal log ottimizzato (prendi l'ULTIMO disponibile)
            rot_mhz = parse_last_rotconst_mhz(log_txt)  # sempre l’ULTIMA occorrenza, in MHz
            # energia ottimizzata (Eh): prima 'SCF Done' se presente, altrimenti ultimo 'HF='
            Eh = extract_last_energy_eh(log_txt)
            m = re.search(r'idx(\d+)', base)
            idx = int(m.group(1)) if m else 10**9
            candidates.append({
                "idx": idx,
                "name": base,
                "xyz": xyz_opt,
                "params": params,
                "rotc": rot_mhz,
                "Eh": Eh,                      # energia assoluta preferita per tie-break
                "dE": de_by_idx.get(idx, float("inf"))  # fallback in assenza di Eh
            })
        except Exception as e:
            print(f"[ERR] {base}: parsing log ottimizzato fallito: {e}")

    if not candidates:
        print("[INFO] Nessun candidato finale disponibile.")
        return

    # dedup finale su rotazionali, TENENDO la più bassa energia (Eh, poi dE)
    def final_sort_key(d):
        Eh = d.get("Eh", None)
        Eh_key = Eh if (Eh is not None and math.isfinite(Eh)) else float('inf')
        return (Eh_key, d.get("dE", float('inf')), d["idx"])

    kept: List[Dict[str, Any]] = []
    for c in sorted(candidates, key=final_sort_key):
        a1,b1,c1 = c["rotc"]
        if not all(math.isfinite(v) for v in (a1,b1,c1)):
            kept.append(c); continue  # senza rotazionali non scartiamo
        dup = False
        for r in kept:
            a2,b2,c2 = r["rotc"]
            if all(math.isfinite(v) for v in (a2,b2,c2)) and rotc_close_3(a1,b1,c1,a2,b2,c2,args.rotc_threshold_pct):
                dup = True
                break
        if not dup:
            kept.append(c)

    print(f"[INFO] Finali: {len(candidates)} candidati -> {len(kept)} unici (rotc pct={args.rotc_threshold_pct}%).")

    # scrivi i NON duplicati
    for c in kept:
        final_gjf = build_final_gjf_from_optimized(gjf_text, c["xyz"], c["params"])
        out_gjf = os.path.join(final_dir, c["name"].replace("opt_", "final_") + ".gjf")
        write_text(out_gjf, final_gjf)
        out_xyz = os.path.join(final_xyz_dir, c["name"].replace("opt_", "final_") + ".xyz")
        write_xyz_file(out_xyz, c["xyz"])
        tagE = f"Eh={c['Eh']:.8f}" if (c.get('Eh') is not None and math.isfinite(c['Eh'])) else f"dE={c.get('dE', float('inf')):.6f}"
        print(f"[OK]  {c['name']}: creati\n       {out_gjf}\n       {out_xyz}   ({tagE})")

if __name__ == "__main__":
    main()
