import os
import csv
import re
import math
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Parametri di configurazione ---
# Example input files are stored in the `examples/` directory.
#INPUT_FILE = "examples/gly.gjf"
#INPUT_FILE = "examples/butano.gjf"
INPUT_FILE = "examples/thiopronine_g16_to_use.gjf"  # file di riferimento fornito dall'utente

TMP_DIR = "tmp"
#GENERATIONS_DIR = "generations_butano"
#GENERATIONS_DIR = "generations_gly"
GENERATIONS_DIR = "generations_tiopronin_new"

NUM_GENERAZIONI = 3
POPOLAZIONE_INIZIALE = 4
POPOLAZIONE_TARGET = 2

# Parametri per penalità similarità
SIMILARITY_THRESHOLD = 0.9  # soglia di similarità oltre la quale si penalizza
CUTOFF_ANGLE = 10*math.sqrt(5)
GAMMA = -math.log(SIMILARITY_THRESHOLD)/(CUTOFF_ANGLE*CUTOFF_ANGLE) # parametro della funzione esponenziale (toonabile)

# --- Parametri aggiuntivi per oscillazione ---
BASE_RATE_MUTATION = 0.6       # Valore di base per crossover e mutazione
BASE_RATE_CROSSOVER = 0.6       # Valore di base per crossover e mutazione
DELTA_RATE_MUTATION = 0.1      # Δ: ampiezza massima dell'oscillazione
DELTA_RATE_CROSSOVER = 0.1      # Δ: ampiezza massima dell'oscillazione
NUM_OSCILLATIONS = 2  # n: numero di ondate da avere durante P generazioni

#Tiopronin
GENI = [
    (3, "D( 15, 3, 2, 11)"),  # chi1
    (3, "D(  3, 2, 4, 5)"),  # psi1
    (3, "D(  4, 6, 7, 8)"),  # phi2
    (3, "D(  6, 7, 8, 9)"),  # psi2
    (2, "D(  9, 8, 10, 19)")  # omega2
]

# --- Funzioni di supporto ---

def read_reference_file(filepath):
    """Legge il file di riferimento e restituisce le righe."""
    with open(filepath, 'r') as f:
        return f.readlines()

def remove_frozen_substring(lines):
    """
    Rimuove la sottostringa "frozen" (e l'eventuale virgola successiva) da ogni riga,
    senza eliminare la riga stessa.
    """
    new_lines = []
    for line in lines:
        new_line = re.sub(r'(?i)frozen,?', '', line)
        new_lines.append(new_line)
    return new_lines

def save_statistics(file_name, generations_data):
    """Scrive le statistiche delle generazioni in un file CSV."""
    fields = ["Generation", "Avg. Fitness", "MAX Fitness", "MIN Fitness", "Mutation rate", "Crossover rate"]
    with open(file_name, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in generations_data:
            writer.writerow(row)

def generate_random_allele(periodicity):
    """Genera un allele casuale in [0,360] (in gradi) secondo la distribuzione:
       P(ang)=b+w*(1+cos(n*ang)), dove ang è in radianti, n=periodicity, w=0.1 e b=1/(2*pi)-w.
       Il campionamento viene effettuato in radianti e il risultato convertito in gradi.
    """
    w = 0.08
    b = 1/(2*math.pi) - w
    M = 1/(2*math.pi) + w  # valore massimo di P(ang)
    n = periodicity       # periodicità
    while True:
        candidate_rad = random.uniform(0, 2*math.pi)
        density = b + w*(1 + math.cos(n * candidate_rad))
        if random.uniform(0, 1) <= density / M:
            return math.degrees(candidate_rad)

def add_gene_lines(lines, geni, alleli):
    """
    Rimuove eventuali righe vuote finali dalla lista di righe,
    aggiunge immediatamente le righe dei geni e infine aggiunge
    una riga vuota extra in fondo.
    """
    # Rimuove le righe vuote in fondo
    while lines and lines[-1].strip() == "":
        lines.pop()
    new_lines = lines.copy()
    # Appende le righe dei geni senza spazi vuoti intermedi
    for idx, ((period, definition), allele) in enumerate(zip(geni, alleli), start=1):
        gene_line = f"GENE{idx}(Value={allele:.4f}) = {definition}\n"
        new_lines.append(gene_line)
    # Aggiunge una riga vuota extra alla fine
    new_lines.append("\n")
    return new_lines

def write_individual_file(directory, individual_id, content_lines):
    """Scrive il file .gjf per l'individuo."""
    filename = os.path.join(directory, f"individuo_{individual_id}.gjf")
    with open(filename, 'w') as f:
        f.writelines(content_lines)
    return filename

def run_gdv(gjf_file, log_file):
    """Esegue il comando 'gdv' sul file gjf e scrive l'output su log_file."""
    with open(gjf_file, 'r') as infile, open(log_file, 'w') as outfile:
        result = subprocess.run(["g16"], stdin=infile, stdout=outfile, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        print(f"Errore nell'esecuzione di g16 per {gjf_file}: {result.stderr}")
    return result.returncode

def parse_fitness(log_file):
    """Estrae l'ultimo valore di fitness cercando 'HF=' nel file log."""
    fitness = None
    with open(log_file, 'r') as f:
        content = f.read()
    matches = re.findall(r"HF=([-+]?\d+\.\d+)", content)
    if matches:
        fitness = float(matches[-1])
    else:
        print(f"Fitness non trovata in {log_file}")
    return fitness

def parse_xyz_from_log(log_file):
    """Estrae il numero di atomi e le coordinate ottimizzate dal log di g16."""
    with open(log_file, 'r') as f:
        content = f.read()
    if "Principal axis orientation:" not in content:
        print(f"Sezione 'Principal axis orientation:' non trovata in {log_file}")
        return None, []

    parts = content.split("Principal axis orientation:")
    after = parts[1]
    sections = after.split(" ---------------------------------------------------------------------")
    if len(sections) < 3:
        print(f"Impossibile individuare la sezione delle coordinate in {log_file}")
        return None, []

    coord_block = sections[-2]
    lines = [line.strip() for line in coord_block.strip().splitlines() if line.strip()]
    if not lines:
        print(f"Nessuna coordinata trovata in {log_file}")
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

def write_xyz_file(directory, individual_id, fitness, num_atoms, xyz_lines):
    filename = os.path.join(directory, f"individuo_{individual_id}.xyz")
    with open(filename, 'w') as f:
        # Scrive il numero di atomi come prima riga
        f.write(f"{num_atoms}\n")
        # Scrive la fitness come seconda riga
        f.write(f"HF={fitness}\n")
        # Scrive ogni riga di coordinate
        for line in xyz_lines:
            f.write(line + "\n")
    return filename

def cleanup_tmp(directory):
    """Elimina tutti i file con estensione .gjf, .log, .chk nella cartella tmp."""
    for fname in os.listdir(directory):
        if fname.endswith((".gjf", ".log", ".chk")):
            os.remove(os.path.join(directory, fname))

def similarity(ind1, ind2, gamma):
    """Calcola la similarità tra due individui usando i loro vettori degli alleli.
       S(A,B)=exp(-gamma * ||A-B||^2)
    """
    diff_sq = sum((a - b) ** 2 for a, b in zip(ind1["alleli"], ind2["alleli"]))
    return math.exp(-gamma * diff_sq)

# --- Funzioni per il Genetic Algorithm ---

def initialize_population(pop_size, geni):
    """Genera la popolazione iniziale come lista di dizionari contenenti:
       - id: identificativo dell'individuo
       - alleli: lista dei valori per ciascun gene
       - fitness: None inizialmente
    """
    population = []
    for i in range(pop_size):
        alleli = [generate_random_allele(period) for period, _ in geni]
        population.append({
            "id": i,
            "alleli": alleli,
            "fitness": None,
            "xyz_file": None
        })
    return population

def evaluate_individual(ind, reference_lines, geni, tmp_dir, gen_dir):
    """Prepara l'input di un individuo, esegue g16 e salva il corrispondente file XYZ."""
    base_lines = remove_frozen_substring(reference_lines)
    individual_lines = add_gene_lines(base_lines, geni, ind["alleli"])
    gjf_file = write_individual_file(tmp_dir, ind["id"], individual_lines)
    log_file = os.path.join(tmp_dir, f"individuo_{ind['id']}.log")

    retcode = run_gdv(gjf_file, log_file)
    if retcode != 0:
        ind["fitness"] = float("inf")
        return

    fitness = parse_fitness(log_file)
    ind["fitness"] = fitness if fitness is not None else float("inf")

    num_atoms, xyz_lines = parse_xyz_from_log(log_file)
    if xyz_lines:
        xyz_file = write_xyz_file(gen_dir, ind["id"], ind["fitness"], num_atoms, xyz_lines)
        ind["xyz_file"] = xyz_file

def selection(population, target_size):
    """Seleziona i target_size individui con fitness migliore (minore)."""
    sorted_pop = sorted(population, key=lambda ind: ind["fitness"])
    return sorted_pop[:target_size]

def crossover(parent1, parent2):
    """Crossover semplice: per ogni allele scegliamo da uno dei due parenti."""
    child_alleles = []
    for a1, a2 in zip(parent1["alleli"], parent2["alleli"]):
        child_alleles.append(a1 if random.random() < 0.5 else a2)
    return child_alleles

def mutate(alleles, mutation_rate, geni):
    """Applica mutazioni casuali sugli alleli."""
    new_alleles = []
    for allele, (period, _) in zip(alleles, geni):
        if random.random() < mutation_rate:
            new_alleles.append(generate_random_allele(period))
        else:
            new_alleles.append(allele)
    return new_alleles

def tournament_selection(population, tournament_size=3):
    """Seleziona il migliore tra un campione casuale di 'tournament_size' individui."""
    competitors = random.sample(population, tournament_size)
    return min(competitors, key=lambda ind: ind["fitness"])

# --- Main Genetic Algorithm ---
def genetic_algorithm():
    """Esegue l'algoritmo genetico completo."""
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(GENERATIONS_DIR, exist_ok=True)
    
    # Prepara un file di log per le statistiche delle generazioni
    generations_data = []
    generation_log_path = os.path.join(GENERATIONS_DIR, "generation_log.txt")
    with open(generation_log_path, "w") as logf:
        logf.write("Log delle generazioni:\n\n")
    
    reference_lines = read_reference_file(INPUT_FILE)
    population = initialize_population(POPOLAZIONE_INIZIALE, GENI)
    
    for gen in range(NUM_GENERAZIONI):
        print(f"Generazione {gen}")
        
        # Calcola i tassi correnti secondo la formula oscillante
        current_mutation_rate = BASE_RATE_MUTATION + DELTA_RATE_MUTATION * math.sin((math.pi * NUM_OSCILLATIONS / (NUM_GENERAZIONI - 1)) * gen)
        current_crossover_rate = BASE_RATE_CROSSOVER - DELTA_RATE_CROSSOVER * math.sin((math.pi * NUM_OSCILLATIONS / (NUM_GENERAZIONI - 1)) * gen)

        gen_dir = os.path.join(GENERATIONS_DIR, f"population_{gen}")
        os.makedirs(gen_dir, exist_ok=True)

        max_workers = min(len(population), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(evaluate_individual, ind, reference_lines, GENI, TMP_DIR, gen_dir)
                for ind in population
            ]
            for future in as_completed(futures):
                future.result()
        
        # Penalizzazione per similarità (abilitare se necessario)
        # penalty_unit = 1e-2 / len(population)
        # for i, ind in enumerate(population):
        #     for j, other in enumerate(population):
        #         if i != j and similarity(ind, other, GAMMA) > SIMILARITY_THRESHOLD:
        #             ind["fitness"] += penalty_unit
        
        # Calcola e logga le statistiche della generazione
        fitness_list = [ind["fitness"] for ind in population if ind["fitness"] is not None]
        if fitness_list:
            avg_fit = sum(fitness_list) / len(fitness_list)
            min_fit = min(fitness_list)
            max_fit = max(fitness_list)
        else:
            avg_fit = min_fit = max_fit = None

        with open(generation_log_path, "a") as logf:
            logf.write(f"Generazione {gen}:\n")
            logf.write(f"  Fitness media: {avg_fit}\n")
            logf.write(f"  Fitness minima: {min_fit}\n")
            logf.write(f"  Fitness massima: {max_fit}\n")
            logf.write(f"  Mutation rate: {current_mutation_rate}\n")
            logf.write(f"  Crossover rate: {current_crossover_rate}\n")
            logf.write("  Fitness degli individui:\n")
            for ind in population:
                logf.write(f"    Individuo {ind['id']} -> Fitness: {ind['fitness']}\n")
            logf.write("\n")
        
        generations_data.append({
            "Generation": gen,
            "Avg. Fitness": avg_fit,
            "MAX Fitness": max_fit,
            "MIN Fitness": min_fit,
            "Mutation rate": current_mutation_rate,
            "Crossover rate": current_crossover_rate
        })

        population = selection(population, POPOLAZIONE_TARGET)
        
        new_population = []
        while len(new_population) < POPOLAZIONE_INIZIALE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            if random.random() < current_crossover_rate:
                child_alleles = crossover(parent1, parent2)
            else:
                child_alleles = parent1["alleli"].copy()
            child_alleles = mutate(child_alleles, current_mutation_rate, GENI)
            new_population.append({
                "id": random.randint(1000, 9999),
                "alleli": child_alleles,
                "fitness": None,
                "xyz_file": None
            })
        
        population = new_population
        #cleanup_tmp(TMP_DIR)

    save_statistics("evolution.csv",generations_data)
    print("Algoritmo completato.")

if __name__ == "__main__":
    genetic_algorithm()


