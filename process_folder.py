#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import copy
import csv

# ================== Utility: lettura XYZ ===================

def read_xyz(file_path):
    """Ritorna (nat, comment, righe_xyz) senza validazione pesante."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if len(lines) < 2:
        raise ValueError(f"File XYZ malformato: {file_path}")
    try:
        nat = int(lines[0].strip())
    except Exception:
        # Se la prima riga non è un intero, proviamo a dedurlo dal contenuto
        nat = sum(1 for ln in lines[2:] if ln.strip())
    comment = lines[1] if len(lines) >= 2 else ""
    geom = [ln for ln in lines[2:2+nat]]
    return nat, comment, geom

# =========== Generazione GJF a partire dall’XYZ ============

def xyz_to_gjf_content(title, geom_lines, nprocshared=4, mem="32Gb",
                       route="#p B3LYP/6-31+G* Opt empiricaldispersion=gd3bj output=pickett geom=gic symm=loose"):
    """
    Costruisce il contenuto del .gjf come richiesto.
    """
    header = []
    header.append(f"%nprocshared={nprocshared}")
    header.append(f"%mem={mem}")
    header.append(route.strip())
    header.append("")  # riga vuota
    header.append(f" {title}")
    header.append("")
    header.append("0 1")
    body = "\n".join(geom_lines)
    return "\n".join(header) + "\n" + body + "\n\n"

# =============== Esecuzione Gaussian =======================

def run_g16(g16_cmd, gjf_path, log_path, env=None):
    """
    Esegue Gaussian: g16 < input.gjf > output.log
    Restituisce il returncode.
    """
    with open(gjf_path, "r") as fin, open(log_path, "w") as fout:
        proc = subprocess.run([g16_cmd], stdin=fin, stdout=fout, stderr=subprocess.PIPE, text=True, env=env)
    if proc.returncode != 0:
        print(f"[ERR] Gaussian fallito su {os.path.basename(gjf_path)}: {proc.stderr.strip()}")
    return proc.returncode

# =========== Parsing log: ultima Standard orientation =======

def parse_last_standard_orientation_xyz(log_path):
    """
    Estrae l’ULTIMA "Standard orientation:" dal log e ritorna (nat, xyz_lines).
    xyz_lines in formato: "<AtomicNumber>  X  Y  Z"
    Se non trovata o no Normal termination, ritorna (None, []).
    """
    try:
        with open(log_path, "r", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        return None, []

    # Check normal termination
    if "Normal termination of Gaussian" not in text:
        return None, []

    key = "Standard orientation:"
    idx = text.rfind(key)
    if idx == -1:
        return None, []

    tail = text[idx:].splitlines()
    # Trova le righe di trattini
    def is_dash(s):
        s = s.strip()
        return bool(s) and set(s) == {"-"}
    dash_idx = [i for i, ln in enumerate(tail) if is_dash(ln)]
    if len(dash_idx) < 2:
        return None, []

    start = dash_idx[1] + 1  # dopo seconda riga di trattini
    xyz_lines = []
    nat = 0
    for ln in tail[start:]:
        if is_dash(ln):
            break
        toks = ln.split()
        # Formato atteso:
        # Ctr  Atomic  Atomic   X          Y          Z
        # Num  Number  Type
        if len(toks) >= 6 and toks[1].isdigit():
            Z = toks[1]
            x = toks[3]; y = toks[4]; z = toks[5]
            xyz_lines.append(f"{Z} {x} {y} {z}")
            nat += 1

    if nat == 0:
        return None, []
    return nat, xyz_lines

# =========== Parsing: energia e costanti rotazionali =======

def parse_energy_from_log(log_file):
    """
    Ritorna l'ULTIMO valore di energia trovato nel .log.
    Priorità:
      1) ultima occorrenza di 'HF=' (archive section)
      2) ultima occorrenza di 'SCF Done: E(...) = ...'
      3) ultima occorrenza generica di 'Energy = ...'
    Gestisce spazi, newline e notazione 'D' per l'esponente.
    """
    try:
        with open(log_file, 'r', errors='ignore') as f:
            text = f.read()
    except FileNotFoundError:
        return None

    # hf_pat = re.compile(r"HF\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)", re.IGNORECASE)

    # dopo (consente H\n F= ...):
    hf_pat = re.compile(
        r"H\s*F\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
        re.IGNORECASE
    )

    hf_matches = list(hf_pat.finditer(text))
    if hf_matches:
        val = hf_matches[-1].group(1).replace('D', 'E').replace('d', 'E')
        try:
            return float(val)
        except ValueError:
            pass

    scf_pat = re.compile(
        r"SCF\s+Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
        re.IGNORECASE
    )
    scf_matches = list(scf_pat.finditer(text))
    if scf_matches:
        val = scf_matches[-1].group(1).replace('D', 'E').replace('d', 'E')
        try:
            return float(val)
        except ValueError:
            pass

    energy_pat = re.compile(
        r"\bEnergy\s*=\s*([-+]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
        re.IGNORECASE
    )
    en_matches = list(energy_pat.finditer(text))
    if en_matches:
        val = en_matches[-1].group(1).replace('D', 'E').replace('d', 'E')
        try:
            return float(val)
        except ValueError:
            pass

    return None

def parse_rotational_constants_mhz(log_file):
    """
    Cerca ' Rotational constants (MHZ):' e legge la riga successiva con A B C.
    Ritorna (A, B, C) come float, oppure (None, None, None) se non trovati.
    """
    A = B = C = None
    try:
        with open(log_file, 'r', errors='ignore') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Rotational constants (MHZ):" in line:
                if i + 1 < len(lines):
                    parts = lines[i + 1].strip().split()
                    if len(parts) >= 3:
                        A = float(parts[0]); B = float(parts[1]); C = float(parts[2])
                break
    except Exception:
        pass
    return A, B, C

def write_xyz(out_path, nat, comment, xyz_lines):
    with open(out_path, "w") as f:
        f.write(f"{nat}\n")
        f.write(comment.strip() + "\n")
        for ln in xyz_lines:
            f.write(ln + "\n")

# ===================== Worker per file =====================

def process_xyz_file(xyz_path, out_dir, g16_cmd, nprocshared, mem, route, env=None, overwrite=False):
    """
    Per un singolo file .xyz:
      - genera .gjf
      - esegue Gaussian
      - se normal termination: estrae ultima Standard orientation e salva <stem>_std.xyz
        includendo nel commento: E=<energia> e A/B/C=<costanti in MHz> se disponibili.
    Ritorna: (stem, success, msg_or_outxyz, E, A, B, C, log_path)
    """
    stem = os.path.splitext(os.path.basename(xyz_path))[0]
    gjf_path = os.path.join(out_dir, f"{stem}.gjf")
    log_path = os.path.join(out_dir, f"{stem}.log")
    out_xyz_path = os.path.join(out_dir, f"{stem}_std.xyz")

    # Se già esistono output, prova a riusare (a meno di --overwrite)
    if (not overwrite) and os.path.exists(out_xyz_path) and os.path.exists(log_path):
        nat, xyz_lines = parse_last_standard_orientation_xyz(log_path)
        if nat:
            E = parse_energy_from_log(log_path)
            A, B, C = parse_rotational_constants_mhz(log_path)
            return (stem, True, out_xyz_path, E, A, B, C, log_path)
        # altrimenti continua (p.es. log senza normal termination)

    # Leggi XYZ
    nat, comment, geom = read_xyz(xyz_path)

    # Crea GJF
    gjf_txt = xyz_to_gjf_content(stem, geom, nprocshared=nprocshared, mem=mem, route=route)
    with open(gjf_path, "w") as f:
        f.write(gjf_txt)

    # Esegui Gaussian
    rc = run_g16(g16_cmd, gjf_path, log_path, env=env)
    if rc != 0:
        return (stem, False, f"[Gaussian rc={rc}]", None, None, None, None, log_path)

    # Estrai ultima Standard orientation se normal termination
    nat2, xyz_lines = parse_last_standard_orientation_xyz(log_path)
    if nat2 and xyz_lines:
        # Nuovo: estrai energia e costanti rotazionali
        E = parse_energy_from_log(log_path)
        A, B, C = parse_rotational_constants_mhz(log_path)

        # Costruisci commento ricco
        parts = [f"{stem} (from {os.path.basename(log_path)})"]
        if E is not None:
            parts.append(f"E={E:.12f}")
        if A is not None and B is not None and C is not None:
            parts.append(f"A={A:.6f}")
            parts.append(f"B={B:.6f}")
            parts.append(f"C={C:.6f}")
        new_comment = "  ".join(parts)

        write_xyz(out_xyz_path, nat2, new_comment, xyz_lines)
        return (stem, True, out_xyz_path, E, A, B, C, log_path)
    else:
        return (stem, False, "[No Normal termination o Standard orientation mancante]", None, None, None, None, log_path)

# =========================== main ===========================

def main():
    ap = argparse.ArgumentParser(description="Batch: XYZ -> GJF, run Gaussian in parallelo, estrai ultima Standard orientation come XYZ, e salva CSV riassuntivo.")
    ap.add_argument("--xyz-dir", required=True, help="Cartella con i file .xyz di input.")
    ap.add_argument("--out-subdir", default="gaussian_runs", help="Nome della sottocartella dove salvare .gjf/.log/.xyz.")
    ap.add_argument("--g16-cmd", default="g16", help="Comando eseguibile per Gaussian (default: g16).")
    ap.add_argument("--cpu-fraction", type=float, default=0.75, help="Frazione di carico CPU per parallelo (0<f<=1).")
    ap.add_argument("--nprocshared", type=int, default=4, help="Valore da inserire in %nprocshared= (default 4).")
    ap.add_argument("--mem", default="32Gb", help="Valore da inserire in %mem= (default 32Gb).")
    ap.add_argument("--route", default="#p B3LYP/6-31+G* Opt empiricaldispersion=gd3bj output=pickett geom=gic symm=loose",
                    help="Riga di route per Gaussian (default: come da richiesta).")
    ap.add_argument("--gauss-scrdir", default=None, help="Se impostato, esporta GAUSS_SCRDIR per i job.")
    ap.add_argument("--overwrite", action="store_true", help="Ricalcola anche se esistono già output.")
    ap.add_argument("--csv-path", default=None, help="Percorso del CSV riassuntivo. Default: <out-subdir>/optimized_summary.csv")
    args = ap.parse_args()

    xyz_dir = args.xyz_dir
    if not os.path.isdir(xyz_dir):
        raise SystemExit(f"Cartella non trovata: {xyz_dir}")

    out_dir = os.path.join(xyz_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = args.csv_path or os.path.join(out_dir, "optimized_summary.csv")

    # Lista file XYZ
    xyz_files = [os.path.join(xyz_dir, f) for f in os.listdir(xyz_dir) if f.lower().endswith(".xyz")]
    xyz_files.sort()
    if not xyz_files:
        raise SystemExit("Nessun file .xyz trovato nella cartella di input.")

    # Calcolo del livello di parallelo: ~ floor(cpu_count / nprocshared) * cpu_fraction
    cpu = os.cpu_count() or 1
    max_jobs_raw = max(1, cpu // max(1, args.nprocshared))
    max_workers = max(1, int(math.floor(max_jobs_raw * max(0.01, min(1.0, args.cpu_fraction)))))

    print(f"[INFO] CPU totali: {cpu} | nprocshared per job: {args.nprocshared} | job concorrenti: {max_workers}")
    print(f"[INFO] Output in: {out_dir}")

    # Env opzionale (GAUSS_SCRDIR)
    env = None
    if args.gauss_scrdir:
        env = copy.deepcopy(os.environ)
        env["GAUSS_SCRDIR"] = args.gauss_scrdir

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for xyz in xyz_files:
            futs.append(ex.submit(
                process_xyz_file,
                xyz, out_dir, args.g16_cmd, args.nprocshared, args.mem, args.route,
                env, args.overwrite
            ))
        for fut in as_completed(futs):
            results.append(fut.result())

    # Report finale
    ok = sum(1 for _, success, *_ in results if success)
    fail = len(results) - ok
    print(f"[DONE] Completati: {ok} | Falliti/Skippati: {fail}")
    for stem, success, msg_or_outxyz, *_rest in results:
        print(f" - {stem}: {'OK' if success else 'FAIL'} -> {msg_or_outxyz}")

    # ====== Scrittura CSV riassuntivo ======
    # Colonne: stem, optimized_xyz, log_path, E (Ha), A_MHz, B_MHz, C_MHz
    rows = []
    for (stem, success, msg_or_outxyz, E, A, B, C, log_path) in results:
        if success:
            rows.append({
                "stem": stem,
                "optimized_xyz": msg_or_outxyz,  # è il path dell'xyz ottimizzato
                "log_path": log_path,
                "E_Ha": f"{E:.12f}" if isinstance(E, float) else "",
                "A_MHz": f"{A:.6f}" if isinstance(A, float) else "",
                "B_MHz": f"{B:.6f}" if isinstance(B, float) else "",
                "C_MHz": f"{C:.6f}" if isinstance(C, float) else "",
            })

    # opzionale: ordina per energia se disponibile
    def _energy_key(r):
        try:
            return float(r["E_Ha"])
        except Exception:
            return float("inf")
    rows.sort(key=_energy_key)

    # scrivi CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=["stem", "optimized_xyz", "log_path", "E_Ha", "A_MHz", "B_MHz", "C_MHz"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[CSV] Salvato riassunto: {csv_path}")

if __name__ == "__main__":
    main()

