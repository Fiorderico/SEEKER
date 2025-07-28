import os
import re
import math
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Parametri di configurazione ---
INPUT_FILE = "input_reference.gjf"  # file di riferimento fornito dall'utente
#INPUT_FILE = "butano.gjf"  # file di riferimento fornito dall'utente
TMP_DIR = "tmp"
#GENERATIONS_DIR = "generations_butano"
GENERATIONS_DIR = "generations"
GENERATIONS_DIR = "generations_gly_prob"
NUM_GENERAZIONI = 30
POPOLAZIONE_INIZIALE = 20
POPOLAZIONE_TARGET = 20
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.5

# Esempio di lista di geni: (periodicità, definizione)
#Gly
GENI = [
    (3, "D(  6,  1,  2,  3)"),
    (3, "D(  1,  2,  3,  4)"),
    (2, "D(  4,  3,  5, 10)")
]

#Butano
#GENI = [
#    (3, "D(5,1,2,3)"),
#    (3, "D(1,2,3,4)"),
#    (3, "D(2,3,4,13)")
#]

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

#def generate_random_allele(periodicity):
#    """Genera un allele casuale in base alla periodicità."""
#    if periodicity == 2:
#        return random.uniform(0, 360)
#    elif periodicity == 3:
#        return random.uniform(0, 360)
#    else:
#        return random.uniform(0, 360/periodicity)

def generate_random_allele(periodicity):
    """Genera un allele casuale in [0,360] (in gradi) secondo la distribuzione:
       P(ang)=b+w*(1+cos(n*ang)), dove ang è in radianti, n=periodicity, w=0.1 e b=1/(2*pi)-w.
       Il campionamento viene effettuato in radianti e il risultato convertito in gradi.
    """
    w = 0.1
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
        gene_line = f"GENE{idx}(frozen,Value={allele:.4f}) = {definition}\n"
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
        result = subprocess.run(["gdv"], stdin=infile, stdout=outfile, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        print(f"Errore nell'esecuzione di gdv per {gjf_file}: {result.stderr}")
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

def parse_xyz_from_log(log_file, fitness):
    with open(log_file, 'r') as f:
        content = f.read()
    # Verifica la presenza della sezione
    if "Principal axis orientation:" not in content:
        print(f"Sezione 'Principal axis orientation:' non trovata in {log_file}")
        return None, []

    # Divide il contenuto in base a "Principal axis orientation:"
    parts = content.split("Principal axis orientation:")
    after = parts[1]

    # Divide la parte successiva usando la linea di separazione
    sections = after.split(" ---------------------------------------------------------------------")
    if len(sections) < 3:
        print(f"Impossibile individuare la sezione delle coordinate in {log_file}")
        return None, []

    # Il penultimo elemento contiene le coordinate
    coord_block = sections[-2]

    # Divide in righe, eliminando quelle vuote
    lines = [line.strip() for line in coord_block.strip().splitlines() if line.strip()]
    if not lines:
        print(f"Nessuna coordinata trovata in {log_file}")
        return None, []

    # Il numero di atomi viene preso dal primo token dell'ultima riga
    try:
        num_atoms = int(lines[-1].split()[0])
    except Exception as e:
        num_atoms = len(lines)

    xyz_lines = []
    for line in lines:
        tokens = line.split()
        # Assicurati che la riga contenga almeno 5 token
        if len(tokens) >= 5:
            # Il primo token è l'indice, il secondo il numero atomico, i successivi le coordinate
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
            "gjf_file": None,
            "xyz_file": None
        })
    return population

def evaluate_individual(ind, reference_lines, geni, tmp_dir):
    # Rimuovi la sottostringa "frozen" dalle righe
    base_lines = remove_frozen_substring(reference_lines)
    # Aggiungi le righe dei geni con i valori attuali degli alleli
    individual_lines = add_gene_lines(base_lines, geni, ind["alleli"])
    # Scrivi il file .gjf
    gjf_file = write_individual_file(tmp_dir, ind["id"], individual_lines)
    log_file = os.path.join(tmp_dir, f"individuo_{ind['id']}.log")

    retcode = run_gdv(gjf_file, log_file)
    if retcode != 0:
        ind["fitness"] = float("inf")
        return

    fitness = parse_fitness(log_file)
    ind["fitness"] = fitness if fitness is not None else float("inf")

    # Ottieni il numero di atomi e le righe xyz
    num_atoms, xyz_lines = parse_xyz_from_log(log_file, ind["fitness"])
    if xyz_lines:
        gen_dir = os.path.join(GENERATIONS_DIR, f"population_gen")
        os.makedirs(gen_dir, exist_ok=True)
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

# --- Main Genetic Algorithm ---
def genetic_algorithm():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(GENERATIONS_DIR, exist_ok=True)
    
    # Prepara un file di log per le statistiche delle generazioni
    generation_log_path = os.path.join(GENERATIONS_DIR, "generation_log.txt")
    with open(generation_log_path, "w") as logf:
        logf.write("Log delle generazioni:\n\n")
    
    reference_lines = read_reference_file(INPUT_FILE)
    population = initialize_population(POPOLAZIONE_INIZIALE, GENI)
    
    for gen in range(NUM_GENERAZIONI):
        print(f"Generazione {gen}")
        # Esecuzione in parallelo per ogni individuo della generazione
        with ThreadPoolExecutor(max_workers=len(population)) as executor:
            futures = [executor.submit(evaluate_individual, ind, reference_lines, GENI, TMP_DIR) for ind in population]
            for future in as_completed(futures):
                future.result()  # Aspetta che ogni conto sia terminato
        
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
            logf.write("  Fitness degli individui:\n")
            for ind in population:
                logf.write(f"    Individuo {ind['id']} -> Fitness: {ind['fitness']}\n")
            logf.write("\n")
        
        population = selection(population, POPOLAZIONE_TARGET)
        
        gen_dir = os.path.join(GENERATIONS_DIR, f"population_{gen}")
        os.makedirs(gen_dir, exist_ok=True)
        for ind in population:
            if ind.get("xyz_file"):
                shutil.copy(ind["xyz_file"], gen_dir)
        
        new_population = []
        while len(new_population) < POPOLAZIONE_INIZIALE:
            parents = random.sample(population, 2)
            if random.random() < CROSSOVER_RATE:
                child_alleles = crossover(parents[0], parents[1])
            else:
                child_alleles = parents[0]["alleli"].copy()
            child_alleles = mutate(child_alleles, MUTATION_RATE, GENI)
            new_population.append({
                "id": random.randint(1000, 9999),
                "alleli": child_alleles,
                "fitness": None,
                "gjf_file": None,
                "xyz_file": None
            })
        
        population = new_population
        cleanup_tmp(TMP_DIR)
    
    print("Algoritmo completato.")

if __name__ == "__main__":
    genetic_algorithm()

