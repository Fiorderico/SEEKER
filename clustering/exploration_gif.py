import os
import re
import subprocess
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import imageio
import io
import shutil
import random

# Per riproducibilità
random.seed(42)
np.random.seed(42)

def create_gjf_from_xyz(xyz_file, dihed_defs, gjf_file):
    with open(xyz_file, 'r') as f:
        xyz_content = f.read().strip()
    filtered_lines = []
    for line in xyz_content.splitlines():
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
    for i, dihed in enumerate(dihed_defs, start=1):
        gjf_lines.append(f"Dihed{i} = {dihed}\n")
    gjf_lines.append("\n")
    with open(gjf_file, 'w') as f:
        f.writelines(gjf_lines)

def run_gdv(gjf_file):
    result = subprocess.run(["gdv", gjf_file],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True)
    return result.returncode

def parse_log_file(log_file, num_dihed):
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
    temp_dir = os.path.join(folder, "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    features = []
    filenames = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".xyz"):
            xyz_file = os.path.join(folder, fname)
            feat, base_name = process_individual(xyz_file, dihed_defs, temp_dir)
            features.append(feat)
            filenames.append(base_name)
    return features, filenames

def normalize_features(features, num_rot=3):
    features = np.array(features)
    rot = features[:, :num_rot]
    min_rot = np.min(rot, axis=0)
    max_rot = np.max(rot, axis=0)
    diff_rot = max_rot - min_rot
    diff_rot[diff_rot == 0] = 1.0
    norm_rot = (rot - min_rot) / diff_rot
    dihed = features[:, num_rot:]
    norm_dihed = (dihed + 180) / 360
    norm_features = np.hstack((norm_rot, norm_dihed))
    return norm_features

def cluster_features(feature_space, distance_threshold=0.2):
    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=distance_threshold,
                                         affinity='euclidean',
                                         linkage='average')
    labels = clustering.fit_predict(feature_space)
    return labels

def get_feature_space_image(feature_space, labels, filenames):
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(feature_space[:, 0], feature_space[:, 1], c=labels, cmap="tab10", s=50)
    for i, fname in enumerate(filenames):
        ax.annotate(fname, (feature_space[i, 0], feature_space[i, 1]), fontsize=8, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Feature Space (PCA 2D) dei Sistemi")
    fig.colorbar(scatter, label="Cluster", ax=ax)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    image = imageio.imread(buf)
    plt.close(fig)
    return image

def main():
    parser = argparse.ArgumentParser(description="Genera una GIF che mostra la PCA (2D) dei sistemi per le cartelle population_i")
    parser.add_argument("root_path", help="Cartella radice contenente le cartelle population_i")
    parser.add_argument("ref_index", type=int, help="Numero totale di cartelle (es. 40)")
    parser.add_argument("--dihedrals", nargs="+", required=True,
                        help="Definizioni dei diedri (es. 'D(1,2,3,4)' 'D(5,6,7,8)' ...)")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Soglia di distanza per il clustering (default: 0.2)")
    args = parser.parse_args()
    
    frames = []
    
    for i in range(args.ref_index):
        pop_folder = os.path.join(args.root_path, f"population_{i}")
        if not os.path.isdir(pop_folder):
            print(f"Cartella {pop_folder} non trovata, salto.")
            continue
        print(f"Processando {pop_folder}...")
        features, filenames = load_population_features(pop_folder, args.dihedrals)
        if not features:
            print(f"Nessun file .xyz trovato in {pop_folder}")
            continue
        features = np.array(features)
        norm_features = normalize_features(features)
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(norm_features)
        labels = cluster_features(features_2d, distance_threshold=args.threshold)
        img = get_feature_space_image(features_2d, labels, filenames)
        frames.append(img)
    
    if not frames:
        print("Nessun frame generato. Controlla i dati.")
        return
    
    output_gif = os.path.join(args.root_path, "population_evolution.gif")
    imageio.mimsave(output_gif, frames, duration=0.5)
    print(f"GIF salvata in {output_gif}")

if __name__ == "__main__":
    main()


