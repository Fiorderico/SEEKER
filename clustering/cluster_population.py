import os
import re
import subprocess
import argparse
import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import shutil
import sys

random.seed(42)
np.random.seed(42)

def create_gjf_from_xyz(xyz_file, dihed_defs, gjf_file):
    """
    Crea un file .gjf a partire da un file .xyz.
    Il file .gjf avrà il seguente formato:
    
    #p UFF GEOM=READALLGIC OUTPUT=PICKETT

     <nome_file_xyz>

    0 1
    <contenuto del file xyz filtrato: solo le righe che hanno 4 valori separati da spazi>
    Dihed1 = <dihed_defs[0]>
    Dihed2 = <dihed_defs[1]>
    ...
    lista_dei_diedri
    """
    with open(xyz_file, 'r') as f:
        xyz_content = f.read().strip()
    
    # Filtra solo le righe che hanno esattamente 4 token
    filtered_lines = []
    for line in xyz_content.splitlines()[2:]:
        tokens = line.strip().split()
        if len(tokens) == 4:
            filtered_lines.append(line)
    
    filtered_xyz = "\n".join(filtered_lines) + "\n"
    
    base_name = os.path.basename(xyz_file)
    
    gjf_lines = []
    gjf_lines.append("#p UFF GEOM=READALLGIC OUTPUT=PICKETT\n")
    gjf_lines.append("\n")
    gjf_lines.append(f"{base_name}\n")
    gjf_lines.append("\n")
    gjf_lines.append("0 1\n")
    gjf_lines.append(filtered_xyz)
    gjf_lines.append("\n")
    # Aggiunge le definizioni dei diedri con nomi assegnati automaticamente
    for i, dihed in enumerate(dihed_defs, start=1):
        gjf_lines.append(f"Dihed{i} = {dihed}\n")
    gjf_lines.append("\n")
    
    with open(gjf_file, 'w') as f:
        f.writelines(gjf_lines)

def run_gdv(gjf_file):
    """
    Lancia il comando 'gdv' sul file gjf.
    Si assume che gdv sia presente nel PATH.
    Attende il completamento e restituisce il codice di ritorno.
    """
    result = subprocess.run(["gdv", gjf_file],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
    return result.returncode

def parse_log_file(log_file, num_dihed):
    """
    Parsifica il file log prodotto da gdv per estrarre le feature:
    - I Rotational constants (MHZ): cerca la sezione "Rotational constants (MHZ):" e ne legge i 3 numeri.
    - I valori dei diedri: per ciascun diedro (Dihed1, Dihed2, …) estrae il valore.
    
    Restituisce un vettore feature: [rot1, rot2, rot3, dihed1, ..., dihedN]
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    rot_pattern = r"Rotational constants \(MHZ\):\s*\n\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
    rot_match = re.search(rot_pattern, content)
    if rot_match:
        rot_consts = [float(rot_match.group(i)) for i in range(1, 4)]
    else:
        print(f"Attenzione: Rotational constants non trovate in {log_file}")
        rot_consts = [0.0, 0.0, 0.0]
    
    dihed_values = []
    for i in range(1, num_dihed+1):
        pattern = rf"!\s*Dihed{i}\s+\S+\s+([-\d\.]+)"
        m = re.search(pattern, content)
        if m:
            dihed_val = float(m.group(1))
        else:
            print(f"Attenzione: Valore per Dihed{i} non trovato in {log_file}")
            dihed_val = 0.0
        dihed_values.append(dihed_val)
    
    feature_vector = rot_consts + dihed_values
    return feature_vector

def process_individual(xyz_file, dihed_defs, temp_dir):
    """
    Processa un singolo individuo:
      1. Crea il file .gjf corrispondente partendo dal file .xyz.
      2. Lancia gdv sul file .gjf.
      3. Parsifica il file .log risultante per ottenere il vettore feature.
    Restituisce il vettore feature e il nome base dell'individuo.
    """
    base_name = os.path.splitext(os.path.basename(xyz_file))[0]
    gjf_file = os.path.join(temp_dir, base_name + ".gjf")
    
    create_gjf_from_xyz(xyz_file, dihed_defs, gjf_file)
    
    retcode = run_gdv(gjf_file)
    if retcode != 0:
        print(f"Errore nell'esecuzione di gdv per {gjf_file}")
    
    log_file = os.path.join(temp_dir, base_name + ".log")
    feature_vector = parse_log_file(log_file, len(dihed_defs))
    return feature_vector, base_name

def load_population_features(folder, dihed_defs):
    """
    Processa tutti i file .xyz nella cartella della popolazione.
    Utilizza una directory temporanea (creata in "folder/temp_processing") per i file intermedi.
    Restituisce:
      - features: lista dei vettori feature
      - filenames: lista dei nomi base degli individui
    """
    temp_dir = os.path.join(folder, "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    
    features = []
    filenames = []
    for fname in os.listdir(folder):
        if fname.endswith(".xyz"):
            xyz_file = os.path.join(folder, fname)
            feat, base_name = process_individual(xyz_file, dihed_defs, temp_dir)
            features.append(feat)
            filenames.append(base_name)
    
    return features, filenames

def normalize_features(features, num_rot=3):
    """
    Normalizza il vettore feature:
      - Le prime num_rot colonne (rotational constants) vengono normalizzate con MinMax scaling (sul dataset corrente).
      - Le colonne relative ai diedri vengono normalizzate da [-180,180] a [0,1] con la formula (x+180)/360.
    """
    features = np.array(features)
    rot = features[:, :num_rot]
    min_rot = np.min(rot, axis=0)
    max_rot = np.max(rot, axis=0)
    diff_rot = max_rot - min_rot
    diff_rot[diff_rot == 0] = 1.0
    norm_rot = (rot - min_rot) / diff_rot
    
    dihed = features[:, num_rot:]
    norm_dihed = (dihed + 180) / 360
    
    #norm_features = np.hstack((norm_rot, norm_dihed))
    norm_features = norm_dihed
    return norm_features

def cluster_features(feature_space, distance_threshold=0.2):
    """
    Esegue l'agglomerative clustering sui vettori (in questo caso nello spazio 2D).
    Il parametro distance_threshold controlla la soglia di unione dei cluster.
    Restituisce le etichette dei cluster.
    """
    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=distance_threshold,
                                         affinity='euclidean',
                                         linkage='average')
    labels = clustering.fit_predict(feature_space)
    return labels

def select_representative(feature_space, labels, filenames):
    """
    Per ciascun cluster, seleziona la struttura rappresentativa come quella con la distanza media minima dagli altri membri.
    Restituisce due dizionari:
      - clusters: {label: [(filename, feature_vector), ...]}
      - representatives: {label: filename_rappresentativo}
    """
    clusters = {}
    for label, fname, feat in zip(labels, filenames, feature_space):
        clusters.setdefault(label, []).append((fname, feat))
    
    representatives = {}
    for label, items in clusters.items():
        if len(items) == 1:
            representatives[label] = items[0][0]
        else:
            feats = np.array([item[1] for item in items])
            dists = np.linalg.norm(feats[:, None] - feats[None, :], axis=2)
            avg_dists = np.mean(dists, axis=1)
            rep_index = np.argmin(avg_dists)
            representatives[label] = items[rep_index][0]
    return clusters, representatives

def plot_feature_space(feature_space, labels, filenames, output="feature_space.png"):
    """
    Salva un'immagine PNG del feature space 2D (già ridotto con PCA) 
    con i punti colorati in base al cluster e annota accanto ogni punto il nome del file.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(feature_space[:, 0], feature_space[:, 1], c=labels, cmap="tab10", s=50)
    
    for i, fname in enumerate(filenames):
        plt.annotate(fname, (feature_space[i, 0], feature_space[i, 1]), fontsize=8, alpha=0.7)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Feature Space (PCA 2D) dei Sistemi")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Immagine salvata in {output}")

def main():
    parser = argparse.ArgumentParser(description="Post-processing della popolazione: clustering delle strutture")
    parser.add_argument("folder", help="Cartella della popolazione da processare (contiene file .xyz)")
    parser.add_argument("--dihedrals", nargs="+", required=True,
                        help="Definizioni dei diedri (es. 'D(1,2,3,4)' 'D(5,6,7,8)' ...)")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Soglia di distanza per il clustering (default: 0.2)")
    args = parser.parse_args()
    
    # Reindirizza i print in un file log
    folder_basename = os.path.basename(os.path.normpath(args.folder))
    log_file_path = f"{folder_basename}_cluster_log.txt"
    log_file = open(log_file_path, "w")
    sys.stdout = log_file

    print("Processo la popolazione nella cartella:", args.folder)
    print("Utilizzo le seguenti definizioni di diedri:", args.dihedrals)
    
    features, filenames = load_population_features(args.folder, args.dihedrals)
    if not features:
        print("Nessun file .xyz trovato o nessuna feature estratta.")
        return
    
    features = np.array(features)
    norm_features = normalize_features(features)
    
    # Riduzione a 2D tramite PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(norm_features)
    
    # Clustering nello spazio 2D
    labels = cluster_features(features_2d, distance_threshold=args.threshold)
    clusters, representatives = select_representative(features_2d, labels, filenames)
    
    print("\nRisultati del clustering:")
    for label, items in clusters.items():
        rep = representatives[label]
        membri = [fname for fname, feat in items]
        print(f"Cluster {label}: Rappresentante: {rep}, Membri: {membri}")
    
    # Salva l'immagine del feature space 2D
    plot_feature_space(features_2d, labels, filenames)
    
    # Crea la cartella per i centri dei cluster e copia i file .xyz dei rappresentanti
    centers_folder = f"icluster_centers_{folder_basename}"
    os.makedirs(centers_folder, exist_ok=True)
    print("\nCopio i file .xyz dei centri dei cluster nella cartella:", centers_folder)
    for rep in representatives.values():
        src = os.path.join(args.folder, rep + ".xyz")
        dest = os.path.join(centers_folder, rep + ".xyz")
        try:
            shutil.copy(src, dest)
            print(f"Copiato {rep}.xyz in {centers_folder}")
        except Exception as e:
            print(f"Errore nella copia di {rep}.xyz: {e}")
    
    log_file.close()

if __name__ == "__main__":
    main()


