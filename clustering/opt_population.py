#!/usr/bin/env python3
import os
import sys
import subprocess
import concurrent.futures
import re

def create_gjf_from_xyz(xyz_path, output_folder):
    """
    Legge un file XYZ e crea il file GJF corrispondente con il formato:
    
    %Nprocshared=12
    %Mem=64GB
    #p B3LYP/6-31G* OPT OUTPUT=PICKETT

    <nome_file_senza_estensione>

    0 1
    <coordinate (linee 3 in poi del file xyz)>
    """
    base_name = os.path.splitext(os.path.basename(xyz_path))[0]
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    # Supponiamo che le prime due righe siano da scartare (numero di atomi e commento)
    geometry = lines[2:]
    
    content = []
    content.append("%Nprocshared=12\n")
    content.append("%Mem=64GB\n")
    content.append("#p B3LYP/6-31G* OPT OUTPUT=PICKETT\n")
    content.append("\n")
    content.append(base_name + "\n")
    content.append("\n")
    content.append("0 1\n")
    content.extend(geometry)
    content.append("\n")
    
    gjf_filename = os.path.join(output_folder, base_name + ".gjf")
    with open(gjf_filename, 'w') as f:
        f.writelines(content)
    return gjf_filename

def run_gdv_on_gjf(gjf_file):
    """
    Esegue il comando 'gdv <gjf_file>' e redirige l'output su un file log
    (lo stesso nome del file GJF, ma con estensione .log).
    """
    log_file = os.path.splitext(gjf_file)[0] + ".log"
    with open(log_file, 'w') as logf:
        result = subprocess.run(["gdv", gjf_file], stdout=logf, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        print(f"Errore nell'esecuzione di gdv per {gjf_file}: {result.stderr}")
    return log_file

def extract_optimized_xyz(log_file):
    """
    Estrae la geometria ottimizzata dal file log.
    Si cerca l'ultimo blocco 'Standard orientation:' e si estraggono le righe di coordinate
    che seguono la seconda linea separatrice ('-----').
    
    Restituisce una lista di stringhe formattate per un file XYZ, oppure None se l'estrazione fallisce.
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Suddividi in base a "Standard orientation:"
    sections = content.split("Standard orientation:")
    if len(sections) < 2:
        print(f"Blocco 'Standard orientation:' non trovato in {log_file}")
        return None
    # Prendi l'ultima occorrenza
    block = sections[-1]
    lines = block.splitlines()
    
    # Trova la prima linea di separazione (contenente "-----")
    start_index = None
    for i, line in enumerate(lines):
        if "-----" in line:
            start_index = i
            break
    if start_index is None:
        print(f"Linea di separazione non trovata in {log_file}")
        return None
    
    # Trova la seconda linea di separazione
    second_dash_index = None
    for i in range(start_index+1, len(lines)):
        if "-----" in lines[i]:
            second_dash_index = i
            break
    if second_dash_index is None:
        print(f"Seconda linea di separazione non trovata in {log_file}")
        return None

    coords = []
    # Le coordinate iniziano dalla linea dopo la seconda linea di separazione
    for line in lines[second_dash_index+1:]:
        if "-----" in line:
            break
        tokens = line.split()
        if len(tokens) < 6:
            continue
        # In Gaussian il formato è: Center  Atomic  Type   X   Y   Z
        # Estraiamo il secondo token (numero atomico) e le coordinate X, Y, Z.
        coords.append(f"{tokens[1]} {tokens[3]} {tokens[4]} {tokens[5]}\n")
    
    if not coords:
        print(f"Nessuna coordinata trovata in {log_file}")
        return None
    
    num_atoms = len(coords)
    xyz_content = [f"{num_atoms}\n", f"Optimized geometry from {os.path.basename(log_file)}\n"]
    xyz_content.extend(coords)
    return xyz_content

def write_xyz_file(xyz_content, base_name, output_folder):
    """
    Scrive un file XYZ con il contenuto della geometria ottimizzata.
    Il file si chiamerà <base_name>_opt.xyz e verrà salvato nella cartella output_folder.
    """
    xyz_filename = os.path.join(output_folder, base_name + "_opt.xyz")
    with open(xyz_filename, 'w') as f:
        f.writelines(xyz_content)
    return xyz_filename

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    if not os.path.isdir(input_folder):
        print(f"La cartella {input_folder} non esiste.")
        sys.exit(1)
    
    # Crea la cartella di output aggiungendo "_gaussian"
    output_folder = input_folder.rstrip("/\\") + "_gaussian"
    os.makedirs(output_folder, exist_ok=True)
    
    # Lista di tutti i file .xyz nella cartella di input
    xyz_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".xyz")]
    gjf_files = []
    
    # Crea i file GJF a partire dagli XYZ
    for xyz_file in xyz_files:
        gjf_file = create_gjf_from_xyz(xyz_file, output_folder)
        gjf_files.append(gjf_file)
    
    # Esegue in parallelo il comando 'gdv' per ogni file GJF
    log_files = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_gjf = {executor.submit(run_gdv_on_gjf, gjf): gjf for gjf in gjf_files}
        for future in concurrent.futures.as_completed(future_to_gjf):
            gjf = future_to_gjf[future]
            log_file = future.result()
            log_files.append(log_file)
    
    # Per ogni file log, estrae l'XYZ ottimizzato e lo salva in un file
    for log_file in log_files:
        xyz_content = extract_optimized_xyz(log_file)
        if xyz_content:
            base_name = os.path.splitext(os.path.basename(log_file))[0]
            xyz_out = write_xyz_file(xyz_content, base_name, output_folder)
            print(f"XYZ ottimizzato salvato in: {xyz_out}")

if __name__ == "__main__":
    main()


