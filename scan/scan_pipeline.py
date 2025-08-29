#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline scan → preselezione → ottimizzazione → finali

DEDUP COMBINATA:
- Rotazionali (MHz): |Ai - Aj|/max(Ai, Aj) <= pct, e idem per B e C (default pct=0.05%)
- + Variabili di SCAN (in gradi): tutte entro |Δ| <= scan_tol_deg con wrap-around per gli angoli
  (default 5.0°). Se una delle due strutture non ha i valori di SCAN disponibili, la dedup usa
  solo il criterio rotazionale (per non perdere unione quando i log non riportano i valori).

Scelte in caso di duplicati: tieni SEMPRE quella a energia più bassa
  * preselezione (prima delle ottimizzazioni): usa ΔE dal summary dello scan
  * finale (dopo le ottimizzazioni): usa l'ultima energia SCF del log (fallback al valore dopo "HF=")

XYZ PER L'OTTIMIZZAZIONE
- per ogni Idx si usa l'XYZ del LOG dello scan (blocchi "Standard orientation")
- mappatura blocchi → Idx: con --pick-blocks first|last (default: first) si decide se usare i primi N blocchi
  o gli ultimi N blocchi trovati nel log dello scan per associarli a Idx = 1..N

ROUTE
- ottimizzazione: "#P geom=readallgic UFF Opt=nomicro Output=Pickett"
- finali:       "#P geom=readallgic UFF Output=Pickett"  (senza Opt)

Output:
- {outdir}/opt_idxXXX.gjf e relativi .log
- {outdir}/FINAL_GJF/final_idxXXX.gjf
- {outdir}/FINAL_XYZ/final_idxXXX.xyz
"""

import argparse, os, re, sys, math, shutil, subprocess, multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

# -------- Regex / numeri ----------
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
            in_summary = True; i += 1; continue

        if in_summary:
            if re.match(r'^\s+\d+(?:\s+\d+)+\s*$', line):
                indices = [int(x) for x in line.split()]
                ncols = len(indices)

                i += 1
                while i < len(lines) and "Eigenvalues --" not in lines[i]:
                    i += 1
                if i >= len(lines): break
                energies = floats_in(lines[i])
                energies = (energies + [float('nan')]*ncols)[:ncols]

                i += 1
                block_params: Dict[str, List[float]] = {}
                while i < len(lines):
                    ln = lines[i]
                    if (re.match(r'^\s+\d+(?:\s+\d+)+\s*$', ln)
                        or "Largest change" in ln or ln.strip() == "" or ln.startswith(" GradGrad")):
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

ROTCONST_LABEL_RE = re.compile(
    r'Rotational constants\s*\(\s*(GHZ|MHZ)\s*\)\s*:',
    re.IGNORECASE
)

def parse_all_rotconst_blocks(text: str) -> List[Tuple[float,float,float]]:
    lines = text.splitlines()
    out: List[Tuple[float,float,float]] = []
    for i, ln in enumerate(lines):
        m = ROTCONST_LABEL_RE.search(ln)
        if not m: 
            continue
        unit = m.group(1).upper()
        nums = floats_in(ln)
        if len(nums) < 3 and i+1 < len(lines): nums += floats_in(lines[i+1])
        if len(nums) < 3 and i+2 < len(lines): nums += floats_in(lines[i+2])
        if len(nums) >= 3:
            a,b,c = nums[:3]
            if unit == 'GHZ': a,b,c = a*1000.0, b*1000.0, c*1000.0
            out.append((a,b,c))
    return out

def parse_last_rotconst_mhz(text: str) -> Tuple[float,float,float]:
    vals = parse_all_rotconst_blocks(text)
    return vals[-1] if vals else (float('nan'),)*3

def rotc_close_3(a1,b1,c1, a2,b2,c2, pct: float) -> bool:
    """Tutte e tre entro pct% rispetto al max componente a coppia."""
    def ok(u,v):
        if not (math.isfinite(u) and math.isfinite(v)): 
            return False
        m = max(abs(u), abs(v))
        if m == 0.0: 
            return abs(u - v) <= 1e-9
        return abs(u - v)/m <= (pct/100.0)
    return ok(a1,a2) and ok(b1,b2) and ok(c1,c2)

# ---------- GIC / GJF helpers ----------

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(path: str, txt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def parse_nprocshared(gjf_text: str, default: int = 8) -> int:
    m = re.search(r'%Nprocshared\s*=\s*(\d+)', gjf_text, flags=re.IGNORECASE)
    return max(1, int(m.group(1))) if m else default

def is_float_token(tok: str) -> bool:
    try: float(tok); return True
    except: return False

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

def extract_sections_from_gjf(gjf_text: str) -> Tuple[str, List[str], List[str]]:
    lines = gjf_text.splitlines()
    idx_charge = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*-?\d+\s+\d+\s*$', ln):
            idx_charge = i; break
    if idx_charge is None:
        raise ValueError("Impossibile individuare carica/molteplicità nel GJF.")

    header = "\n".join(lines[:idx_charge + 1]) + "\n"

    k = idx_charge + 1
    while k < len(lines) and lines[k].strip() == "": k += 1
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
    while g < len(lines) and lines[g].strip() == "": g += 1
    if g < len(lines) and not looks_like_gic_line(lines[g]):
        fallback = None
        for t in range(i, len(lines)):
            if looks_like_gic_line(lines[t]): fallback = t; break
        g = fallback if fallback is not None else len(lines)
    gic_lines = lines[g:] if g is not None and g < len(lines) else []
    return header, xyz_lines, gic_lines

def _join_xyz_and_gic_with_single_blank(xyz_lines, gic_lines) -> str:
    xyz = [ln for ln in xyz_lines if ln.strip() != ""]
    gic = list(gic_lines)
    while gic and gic[0].strip() == "": gic.pop(0)
    xyz_txt = "\n".join(xyz) + "\n\n"
    gic_txt = ("\n".join(gic) + "\n") if gic else "\n"
    return xyz_txt + gic_txt

def patch_route_for_optimization(header: str) -> str:
    def _rewrite_route_line(line: str) -> str:
        line = re.sub(r'(?i)\bgeom\s*=\s*\([^)]*\)|\bgeom\s*=\s*\S+', 'geom=readallgic', line)
        line = re.sub(r'(?i),?\s*modredundant\b', '', line)
        if re.search(r'(?i)\bOpt\b', line):
            line = re.sub(r'(?i)\bOpt\s*(=\s*\([^)]*\)|=\s*\S+)?', 'Opt=nomicro', line)
        else:
            if re.search(r'(?i)\bOutput=Pickett\b', line):
                line = re.sub(r'(?i)\bOutput=Pickett\b', 'Opt=nomicro Output=Pickett', line, count=1)
            else:
                line = line.rstrip() + ' Opt=nomicro'
        line = re.sub(r'[ \t]+', ' ', line)
        return line
    header2 = re.sub(r'(?m)^(#.*)$', lambda m: _rewrite_route_line(m.group(1)), header, count=1)
    return header2

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

def last_standard_orientation_xyz(log_text: str) -> List[Tuple[int, float, float, float]]:
    lines = log_text.splitlines()
    idxs = [i for i, ln in enumerate(lines) if re.search(r'Standard orientation', ln, flags=re.IGNORECASE)]
    if not idxs:
        return []
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
            except: pass
        i += 1
    return pts

def parse_optimized_parameters(log_text: str) -> Dict[str, float]:
    lines = log_text.splitlines()
    heads = [i for i, ln in enumerate(lines) if re.search(r'Optimized Parameters', ln, flags=re.IGNORECASE)]
    if not heads:
        return {}
    i0 = heads[-1]
    i = i0
    while i < len(lines) and 'Name' not in lines[i]: i += 1
    i += 2
    vals: Dict[str, float] = {}
    while i < len(lines):
        ln = lines[i]
        if re.match(r'^\s*-{3,}\s*$', ln): break
        if ln.strip().startswith('!'):
            mname = re.match(r'!\s*([A-Za-z]{4}\d{4})\b', ln)
            if mname:
                name = mname.group(1)
                seg = ln
                seg = seg.split('-DE/DX')[0] if '-DE/DX' in seg else seg
                seg = seg.rsplit('!', 1)[0]
                nums = FLOAT_RE.findall(seg)
                if nums:
                    vals[name] = float(nums[-1].replace('D','E').replace('d','e'))
        i += 1
    return vals

# --- energia finale (Eh)
SCF_DONE_RE = re.compile(r'SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)', re.IGNORECASE)
HF_RE       = re.compile(r'HF=\s*([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)')

def extract_last_energy_eh(log_text: str):
    scf_all = [float(m.group(1)) for m in SCF_DONE_RE.finditer(log_text)]
    if scf_all: return scf_all[-1]
    hf_all = [float(m.group(1)) for m in HF_RE.finditer(log_text)]
    if hf_all: return hf_all[-1]
    return None

def has_normal_termination(log_text: str) -> bool:
    return "Normal termination" in log_text

# --- atomic number → symbol
Z2SYM = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F", 16:"S", 17:"Cl", 35:"Br", 53:"I"}
def atomic_symbol(Z: int) -> str:
    return Z2SYM.get(Z, str(Z))

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

# ---------- Dedup combinata ----------

def scans_close(sc1: Dict[str,float], sc2: Dict[str,float], keys: List[str], tol_deg: float) -> bool:
    """Tutte le ScanXXXX in 'keys' devono essere entro tol_deg (con wrap).
       Se 'keys' è vuoto (o assenza valori in una delle due), ritorna True (non blocca la dedup su rotazionali)."""
    if not keys:
        return True
    for k in keys:
        v1 = sc1.get(k, None); v2 = sc2.get(k, None)
        if v1 is None or v2 is None or not (math.isfinite(v1) and math.isfinite(v2)):
            return False
        if circ_delta_deg(v1, v2) > tol_deg:
            return False
    return True

def dedup_preselection_combined(structs: List[Dict[str, Any]],
                                idx_to_rotc: Dict[int, Tuple[float,float,float]],
                                scan_order: List[str],
                                rotc_pct: float,
                                scan_tol_deg: float) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    sorted_structs = sorted(structs, key=lambda d: (d.get("dE", float('inf')), d["index"]))
    for s in sorted_structs:
        idx = s["index"]
        rotc = idx_to_rotc.get(idx, None)
        sc_vals = s.get("scans", {})
        dup = False
        for r in kept:
            r_rot = idx_to_rotc.get(r["index"], None)
            if rotc is None or r_rot is None:
                continue  # senza rotazionali, non forziamo unione
            # controllo rotazionali
            if not rotc_close_3(*rotc, *r_rot, rotc_pct):
                continue
            # controllo scan sulle chiavi comuni
            common_keys = [k for k in scan_order if (k in sc_vals and k in r.get("scans", {}))]
            if scans_close(sc_vals, r.get("scans", {}), common_keys, scan_tol_deg):
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept

def dedup_final_combined(cands: List[Dict[str, Any]],
                         rotc_pct: float,
                         scan_tol_deg: float,
                         scan_order: List[str]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    # ordina per energia Eh salita; se Eh mancante, usa dE come fallback
    def energy_key(c):
        Eh = c.get("Eh", None)
        return (Eh if (Eh is not None and math.isfinite(Eh)) else float('inf'), c["name"])
    for c in sorted(cands, key=energy_key):
        rotc = c.get("rotc", (float('nan'),)*3)
        sc_vals = {k: v for k, v in c.get("params", {}).items() if k.startswith("Scan")}
        dup = False
        for r in kept:
            rot2 = r.get("rotc", (float('nan'),)*3)
            if not rotc_close_3(*rotc, *rot2, rotc_pct):
                continue
            # scans comuni: se esistono, devono essere entro soglia; se non ci sono, non bloccano
            sc2 = {k: v for k, v in r.get("params", {}).items() if k.startswith("Scan")}
            common_keys = [k for k in scan_order if (k in sc_vals and k in sc2)]
            if scans_close(sc_vals, sc2, common_keys, scan_tol_deg):
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept

# ---------- Build final GJF ----------

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
    header, _, gic_lines = extract_sections_from_gjf(original_gjf_text)
    header = patch_route_for_final(header)
    xyz_block = [f"{Z:>2d} {x:14.6f} {y:12.6f} {z:12.6f}" for Z, x, y, z in xyz_opt]
    def rewrite_gic_line(ln: str) -> str:
        m = re.match(r'^(\s*)([A-Za-z]{4}\d{4})\s*\([^)]*\)\s*(=\s*.*)$', ln)
        if not m:
            m2 = re.match(r'^(\s*)([A-Za-z]{4}\d{4})\s*(=\s*.*)$', ln)
            if not m2: return ln
            prefix, name, tail = m2.groups()
        else:
            prefix, name, tail = m.groups()
        val = opt_params.get(name, None)
        if val is None:
            mval = re.search(r'\(\s*[^)]*Value\s*=\s*([\-\d\.Ee\+]+)', ln, flags=re.IGNORECASE)
            if mval:
                try: val = float(mval.group(1).replace('D','E').replace('d','e'))
                except: val = None
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return ln
        return f"{prefix}{name}(Frozen,Value={val:.6f}) {tail}"
    final_gic_lines = [rewrite_gic_line(ln) for ln in gic_lines]
    while final_gic_lines and final_gic_lines[0].strip() == "": final_gic_lines.pop(0)
    body = _join_xyz_and_gic_with_single_blank(xyz_block, final_gic_lines)
    return header + body

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Pipeline scan + ottimizzazione + dedup combinata (rotazionali + scan).")
    ap.add_argument("--gjf", required=True, help="GJF originale (quello dello scan iniziale).")
    ap.add_argument("--log", required=True, help="LOG risultante dallo scan iniziale (Summary + Standard orientation).")
    ap.add_argument("--de-threshold", type=float, default=0.050, help="Soglia ΔE = E - Emin (stesse unità di Eigenvalues).")
    ap.add_argument("--rotc-threshold-pct", type=float, default=0.05, help="Soglia dup su A,B,C in percentuale (MHz). E.g., 0.05 = 0.05%.")
    ap.add_argument("--scan-tol-deg", type=float, default=5.0, help="Soglia dup per ciascuna ScanXXXX (gradi, wrap).")
    ap.add_argument("--outdir", default="OPT_JOBS", help="Cartella output per job, finali e XYZ.")
    ap.add_argument("--gaussian-exec", default="g16", help="Eseguibile Gaussian (default: g16).")
    ap.add_argument("--run-optimization", action="store_true", help="Esegue anche la fase B (ottimizzazione + finali).")
    ap.add_argument("--pick-blocks", choices=["first","last"], default="first", help="Associazione blocchi 'Standard orientation' del LOG di scan agli indici (default: first).")
    args = ap.parse_args()

    gjf_text = read_text(args.gjf)
    scan_log_text = read_text(args.log)

    # --- A) Parsing summary ---
    structs, scan_order = parse_scan_summary(scan_log_text)
    if not structs:
        print("ERRORE: nessuna struttura trovata nella Summary dello scan.", file=sys.stderr); sys.exit(1)

    # Tabella completa
    def print_full_table(structs: List[Dict[str, Any]], scan_order: List[str]) -> None:
        header = ["Idx", "Energy"] + scan_order
        fmt = "{:>4}  {:>12.6f}  " + "  ".join(["{:>12.6f}"] * len(scan_order))
        print("\n=== TUTTE LE STRUTTURE (summary scan) ===")
        print("  ".join(f"{h:>12}" if i else f"{h:>4}" for i, h in enumerate(header)))
        for s in structs:
            row = [s["index"], s["energy"]] + [s["scans"].get(k, float('nan')) for k in scan_order]
            try: print(fmt.format(*row))
            except: 
                row2 = [row[0], float(row[1])] + [float(x) if isinstance(x,(int,float)) else float('nan') for x in row[2:]]
                print(fmt.format(*row2))

    print_full_table(structs, scan_order)

    # Rotazionali per Idx dal LOG dello scan (blocchi Standard orientation)
    # -> prendi N = numero strutture dalla Summary, associa ai primi/ultimi N blocchi
    all_xyz_blocks = []
    lines = scan_log_text.splitlines()
    pos = [i for i, ln in enumerate(lines) if re.search(r'Standard orientation', ln, flags=re.IGNORECASE)]
    for start in pos:
        i = start+1
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]): i += 1
        if i >= len(lines): break
        i += 1
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]): i += 1
        if i >= len(lines): break
        i += 1
        block_lines = []
        while i < len(lines) and not re.match(r'^\s*-{3,}\s*$', lines[i]):
            block_lines.append(lines[i]); i += 1
        all_xyz_blocks.append(block_lines)
    print(f"[INFO] Trovati {len(all_xyz_blocks)} blocchi 'Standard orientation' nel LOG di scan.")

    # Per ogni blocco, cerca il PRIMO 'Rotational constants' che segue quel blocco
    def rot_after_block(start_line_idx: int) -> Tuple[float,float,float]:
        for j in range(start_line_idx, len(lines)):
            if ROTCONST_LABEL_RE.search(lines[j]):
                # raccogli numeri su questa/successive
                nums = floats_in(lines[j])
                if len(nums) < 3 and j+1 < len(lines): nums += floats_in(lines[j+1])
                if len(nums) < 3 and j+2 < len(lines): nums += floats_in(lines[j+2])
                if len(nums) >= 3:
                    a,b,c = nums[:3]
                    unit = ROTCONST_LABEL_RE.search(lines[j]).group(1).upper()
                    if unit == 'GHZ': a,b,c = a*1000.0, b*1000.0, c*1000.0
                    return (a,b,c)
        return (float('nan'),)*3

    blocks_with_pos = []
    for p in pos:
        blocks_with_pos.append(p)
    # seleziona i primi/ultimi N blocchi
    N = len(structs)
    if args.pick_blocks == "first":
        sel_positions = blocks_with_pos[:N]
    else:
        sel_positions = blocks_with_pos[-N:]
    idx_to_rot = {}
    for k, p0 in enumerate(sel_positions, start=1):
        a,b,c = rot_after_block(p0)
        idx_to_rot[k] = (a,b,c)

    # ΔE filter
    energies = [s["energy"] for s in structs]; emin = min(energies)
    filtered = []
    for s in structs:
        de = s["energy"] - emin
        if de <= args.de_threshold:
            t = dict(s); t["dE"] = de; filtered.append(t)
    # Dedup combinata (rotazionali + scans)
    survivors = dedup_preselection_combined(filtered, idx_to_rot, scan_order, args.rotc_threshold_pct, args.scan_tol_deg)

    # Stampa selezionati
    print("\n=== SELEZIONATI dopo filtro ΔE e dedup (rotazionali + scan) ===")
    print(f"{'Idx':>4}  {'dE':>12}  {'A(MHz)':>12}  {'B(MHz)':>12}  {'C(MHz)':>12}")
    for s in survivors:
        idx = s["index"]; rot = idx_to_rot.get(idx, (float('nan'),)*3)
        print(f"{idx:>4}  {s['dE']:12.6f}  {rot[0]:12.6f}  {rot[1]:12.6f}  {rot[2]:12.6f}")

    if not args.run_optimization:
        return

    # --- B) Ottimizzazione in batch ---
    nprocshared = parse_nprocshared(gjf_text, default=8)
    max_workers = compute_max_workers(nprocshared)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n[INFO] Nprocshared={nprocshared} -> max_workers={max_workers}")
    print(f"[INFO] Genero e lancio {len(survivors)} job in {args.outdir}/")

    # Mappa Idx -> XYZ (dai blocchi dello scan) per usare geometrie specifiche
    idx_to_xyz = {}
    # Ricostruisci le coordinate (Z,x,y,z) dai blocchi selezionati
    def parse_block_xyz(block_lines):
        out = []
        for ln in block_lines:
            parts = ln.split()
            if len(parts) >= 6:
                try:
                    Z = int(parts[1]); x = float(parts[3]); y = float(parts[4]); z = float(parts[5])
                    out.append((Z,x,y,z))
                except: pass
        return out

    selected_blocks = all_xyz_blocks[:N] if args.pick_blocks == "first" else all_xyz_blocks[-N:]
    for idx, block in enumerate(selected_blocks, start=1):
        idx_to_xyz[idx] = parse_block_xyz(block)

    # 1) genera i .gjf per ottimizzazione
    jobs = []
    for s in survivors:
        idx = s["index"]
        scan_values = s["scans"]
        xyz_override = idx_to_xyz.get(idx)
        if xyz_override:
            try:
                _, xyz_lines_ref, _ = extract_sections_from_gjf(gjf_text)
                if xyz_lines_ref and abs(len(xyz_lines_ref) - len(xyz_override)) > 3:
                    print(f"[WARN] idx={idx}: #atomi override ({len(xyz_override)}) molto diverso dal GJF base ({len(xyz_lines_ref)}). Proseguo comunque.")
            except Exception:
                pass
        gjf_opt_text = build_opt_gjf(gjf_text, scan_values, xyz_override=xyz_override)
        gjf_name = f"opt_idx{idx:03d}.gjf"; log_name = f"opt_idx{idx:03d}.log"
        gjf_path = os.path.join(args.outdir, gjf_name); log_path = os.path.join(args.outdir, log_name)
        write_text(gjf_path, gjf_opt_text); jobs.append((gjf_path, log_path))

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2job = {ex.submit(run_gaussian, args.gaussian_exec, gjf, log): (gjf, log) for gjf, log in jobs}
        for fut in as_completed(fut2job):
            gjf_path, log_path = fut2job[fut]
            rc = fut.result(); results[gjf_path] = (log_path, rc)
            print(f"[DONE] {os.path.basename(gjf_path)} -> rc={rc}")

    # 3) finali: parse log, filtro normal termination, raccogli Eh/rotc/params/xyz
    final_dir = os.path.join(args.outdir, "FINAL_GJF")
    final_xyz_dir = os.path.join(args.outdir, "FINAL_XYZ")
    os.makedirs(final_dir, exist_ok=True); os.makedirs(final_xyz_dir, exist_ok=True)

    candidates = []
    for gjf_path, (log_path, rc) in results.items():
        base = os.path.splitext(os.path.basename(gjf_path))[0]
        log_txt = read_text(log_path)
        if not has_normal_termination(log_txt):
            print(f"[WARN] {base}: ottimizzazione NON terminata normalmente."); continue
        try:
            xyz_opt = last_standard_orientation_xyz(log_txt)
            params = parse_optimized_parameters(log_txt)
            rot_mhz = parse_last_rotconst_mhz(log_txt)  # ultima terna, in MHz
            Eh = extract_last_energy_eh(log_txt)
            # prova a recuperare l'Idx dalla stringa file per report
            m = re.search(r'opt_idx(\d+)', base); idx = int(m.group(1)) if m else -1
            candidates.append({"name": base, "idx": idx, "xyz": xyz_opt, "params": params, "rotc": rot_mhz, "Eh": Eh})
        except Exception as e:
            print(f"[ERR] {base}: parsing log ottimizzato fallito: {e}")

    kept = dedup_final_combined(candidates, args.rotc_threshold_pct, args.scan_tol_deg, scan_order)
    print(f"[INFO] Finali: {len(candidates)} candidati -> {len(kept)} unici (rotc pct={args.rotc_threshold_pct}%, scan tol={args.scan_tol_deg}°).")

    for c in kept:
        final_gjf = build_final_gjf_from_optimized(gjf_text, c["xyz"], c["params"])
        out_gjf = os.path.join(final_dir, c["name"].replace("opt_", "final_") + ".gjf")
        write_text(out_gjf, final_gjf)
        out_xyz = os.path.join(final_xyz_dir, c["name"].replace("opt_", "final_") + ".xyz")
        write_xyz_file(out_xyz, c["xyz"])
        tagE = f"Eh={c['Eh']:.8f}" if (c.get('Eh') is not None and math.isfinite(c['Eh'])) else "Eh=NA"
        print(f"[OK]  {c['name']}: creati\n       {out_gjf}\n       {out_xyz}   ({tagE})")

if __name__ == "__main__":
    main()

