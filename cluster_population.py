import os
import argparse
import shutil
import subprocess
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from main import parse_xyz_from_log

# Definition of genes (same as used in genetic algorithm)
GENI = [
    (3, "D( 15, 3, 2, 11)"),
    (3, "D(  3, 2, 4, 5)"),
    (3, "D(  4, 6, 7, 8)"),
    (3, "D(  6, 7, 8, 9)"),
    (2, "D(  9, 8, 10, 19)")
]

# Covalent radii in angstrom (approximate)
COVALENT_RADII = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'S': 1.05,
}

ELEMENT_COLORS = {
    'H': 'white',
    'C': 'black',
    'N': 'blue',
    'O': 'red',
    'S': 'yellow'
}

def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    atoms = []
    coords = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:]])
    return atoms, np.array(coords)

def parse_dihedral(def_str):
    nums = [int(x) for x in def_str.strip('D() ').split(',')]
    return [n - 1 for n in nums]  # convert to 0-based

def dihedral(coord, i, j, k, l):
    p0, p1, p2, p3 = coord[[i, j, k, l]]
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def compute_features(folder):
    features = []
    names = []
    diheds = [parse_dihedral(d) for _, d in GENI]
    for file in sorted(os.listdir(folder)):
        if file.endswith('.xyz'):
            atoms, coord = read_xyz(os.path.join(folder, file))
            angles = [dihedral(coord, *idxs) for idxs in diheds]
            features.append(angles)
            names.append(os.path.splitext(file)[0])
    return np.array(features), names

def draw_molecule(atoms, coord, name, pdf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (a, (x, y, z)) in zip(atoms, coord):
        ax.scatter(x, y, z, color=ELEMENT_COLORS.get(a, 'gray'), s=60)
        ax.text(x, y, z, a)
    n = len(atoms)
    for i in range(n):
        for j in range(i+1, n):
            r1 = COVALENT_RADII.get(atoms[i], 0)
            r2 = COVALENT_RADII.get(atoms[j], 0)
            if np.linalg.norm(coord[i]-coord[j]) <= r1 + r2 + 0.4:
                ax.plot([coord[i,0], coord[j,0]],
                        [coord[i,1], coord[j,1]],
                        [coord[i,2], coord[j,2]], color='gray')
    ax.set_title(name)
    ax.set_axis_off()
    pdf.savefig(fig)
    plt.close(fig)

def select_representatives(features, labels, names, folder):
    rep_folder = os.path.join(folder, 'cluster_representatives')
    os.makedirs(rep_folder, exist_ok=True)
    representatives = {}
    for label in np.unique(labels):
        idx = np.where(labels==label)[0]
        cluster_feats = features[idx]
        center = cluster_feats.mean(axis=0)
        dists = np.linalg.norm(cluster_feats - center, axis=1)
        rep_idx = idx[np.argmin(dists)]
        rep_name = names[rep_idx]
        representatives[label] = rep_name
        shutil.copy(os.path.join(folder, rep_name + '.xyz'),
                    os.path.join(rep_folder, rep_name + '.xyz'))
    return representatives


def read_template(path):
    with open(path) as f:
        return f.readlines()


def build_gaussian_input(template_lines, xyz_file):
    lines = template_lines.copy()
    # ensure Opt keyword
    for i, line in enumerate(lines):
        if line.lstrip().startswith('#'):
            if 'Opt' not in line:
                lines[i] = line.rstrip() + ' Opt\n'
            break
    # find coordinate block
    try:
        charge_idx = next(i for i, l in enumerate(lines) if l.strip().startswith('0'))
    except StopIteration:
        raise ValueError('Charge/multiplicity line not found in template')
    start = charge_idx + 1
    end = start
    while end < len(lines) and lines[end].strip():
        end += 1
    atoms, coord = read_xyz(xyz_file)
    xyz_lines = [f" {a} {x:>10.6f} {y:>10.6f} {z:>10.6f}\n" for a, (x, y, z) in zip(atoms, coord)]
    lines[start:end] = xyz_lines
    return lines


def run_gaussian(gjf_file, log_file):
    try:
        with open(gjf_file, 'r') as inp, open(log_file, 'w') as out:
            res = subprocess.run(['g16'], stdin=inp, stdout=out, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f'Gaussian run failed for {gjf_file}:', res.stderr)
    except FileNotFoundError:
        print('Gaussian executable not found; skipping optimisation.')


def optimize_representatives(reps, folder, template):
    template_lines = read_template(template)
    rep_folder = os.path.join(folder, 'cluster_representatives')
    opt_folder = os.path.join(rep_folder, 'gaussian_opt')
    os.makedirs(opt_folder, exist_ok=True)
    for name in reps.values():
        xyz_path = os.path.join(rep_folder, name + '.xyz')
        gjf_path = os.path.join(opt_folder, name + '.gjf')
        log_path = os.path.join(opt_folder, name + '.log')
        lines = build_gaussian_input(template_lines, xyz_path)
        with open(gjf_path, 'w') as f:
            f.writelines(lines)
        run_gaussian(gjf_path, log_path)
        if os.path.exists(log_path):
            num_atoms, xyz_lines = parse_xyz_from_log(log_path)
            if num_atoms:
                xyz_out = os.path.join(opt_folder, name + '_opt.xyz')
                with open(xyz_out, 'w') as f:
                    f.write(f"{num_atoms}\n\n")
                    for line in xyz_lines:
                        f.write(line + '\n')

def main():
    parser = argparse.ArgumentParser(description='Agglomerative clustering of a population of XYZ molecules.')
    parser.add_argument('population', help='Path to population folder containing XYZ files')
    parser.add_argument('--clusters', type=int, default=2, help='Number of clusters to build')
    parser.add_argument('--pdf', action='store_true', help='Generate a PDF with 3D structures')
    parser.add_argument('--template', help='Gaussian input template to optimise representatives')
    args = parser.parse_args()

    feats, names = compute_features(args.population)
    if feats.size == 0:
        print('No XYZ files found in', args.population)
        return
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    clustering = AgglomerativeClustering(n_clusters=args.clusters)
    labels = clustering.fit_predict(feats_scaled)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(feats_scaled)
    plt.figure()
    scatter = plt.scatter(proj[:,0], proj[:,1], c=labels, cmap='tab10')
    for i, name in enumerate(names):
        plt.annotate(name, (proj[i,0], proj[i,1]), fontsize=8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of population')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(args.population, 'pca_clusters.png'), dpi=300)
    plt.close()

    reps = select_representatives(feats_scaled, labels, names, args.population)
    print('Cluster representatives:', reps)

    if args.template:
        optimize_representatives(reps, args.population, args.template)

    if args.pdf:
        pdf_path = os.path.join(args.population, 'structures.pdf')
        with PdfPages(pdf_path) as pdf:
            for file in sorted(os.listdir(args.population)):
                if file.endswith('.xyz'):
                    atoms, coord = read_xyz(os.path.join(args.population, file))
                    draw_molecule(atoms, coord, os.path.splitext(file)[0], pdf)
        print('Saved PDF to', pdf_path)

if __name__ == '__main__':
    main()
