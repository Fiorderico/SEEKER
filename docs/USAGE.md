# SEEKER usage guide

## 1. Install and configure

```bash
python3 --version  # must report 3.10 or newer
python3 -c 'import sys; assert sys.version_info >= (3,10), sys.version'
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[discovery]'
seeker configure
seeker doctor
```

The `python3` provided by Apple/Xcode may still be Python 3.9 and cannot run
SEEKER. On macOS, install a supported interpreter explicitly when needed:

```bash
brew install python@3.13
"$(brew --prefix python@3.13)/bin/python3.13" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[discovery]'
```

After activation, use `python -m pip`, not an unqualified `pip`, so packages are
installed into the new environment.

`seeker configure` discovers executables in `PATH` and lets each optional
integration be accepted, replaced with an absolute path, or skipped. It writes
`.seeker.local.toml`, which is ignored by Git. Re-run the command whenever a
tool moves. An installed wheel uses `$XDG_CONFIG_HOME/seeker/config.toml` (or
`~/.config/seeker/config.toml`) instead. `SEEKER_CONFIG` selects an explicit
configuration file in either case.

Resolution order is explicit command-line option or environment variable,
local configuration, then `PATH`. Relevant environment variables are
`XTB_COMMAND`, `GAUSSIAN_COMMAND`, and `OPENBABEL_COMMAND`.

## 2. Prepare input

The XYZ file is standard Cartesian XYZ. `GENES.txt` uses one-based atom
indices. An acyclic torsion is written as:

```text
GENE0001(periodicity=3) = D(1,2,3,4)
```

A rigid two-fragment pose is declared as:

```text
POSE_WATER(distance=2.6:5.2,direction=180,orientation=118) = FRAGMENTS(1-9;10-12)
```

Validate before spending electronic-structure time:

```bash
seeker run --xyz INPUT.xyz --genes GENES.txt \
  --output runs/validation --validate-only
```

POSE validation stays inside SEEKER and checks the disconnected-fragment
topology, centroid displacement, relative rotation, fragment rigidity, and
frozen pose frames. Ring-puckering and mixed torsion/pose inputs are rejected.

## 3. Launch a search

Use `seeker launch` or `./run_seeker.sh`. On first use, the launcher starts
the machine setup wizard. It proposes a timestamped directory under `runs/`,
writes every explicit command to `launch_config.txt`, and then uses the same
public interfaces available for scripts. The guided path applies a
coordinate-family search preset and asks only the essential questions. Select
the advanced settings to edit SCAN/LHS/maximin initialization, migration and
operators, geometric filters, optional objectives, early stopping, clustering,
backend parameters, and extended output.

Coordinate realization is always native: acyclic dihedrals use quaternion
rotations and pose-only rigid-fragment inputs use the vectorized SE(3) kernel.
POSE accepts `random` or `maximin` initialization. An xTB
launch can continue automatically through post-clustering optimization and
final RMSD clustering. Both the search and optimization stages resolve xTB from CLI or
environment overrides, then `.seeker.local.toml`, then `PATH`.

Before asking for confirmation, the launcher reports the internal preset
branch, population per island, islands, generations, and nominal upper bound
on energy evaluations. The bound is
`islands × (population + generations × offspring)`; cache hits, rejected
geometries, and deduplication can reduce actual backend calls.

Foreground execution shows the live dashboard. Background execution stores
the detached process ID in `run.pid` and all output in `run.log`; follow it with
`tail -f runs/MY_RUN/run.log`. The review screen is shown before the directory
is created.

For a reproducible non-interactive run, call the CLI directly and record the
seed, coordinate definitions, backend, charge, multiplicity, population per
island, offspring per generation, and number of islands. With multiple islands,
population and offspring counts apply independently to every island.

To favor an N/O/S-H donor directed at the center of a formal double bond, add
the optional `hbond_=` objective:

```bash
seeker run --xyz INPUT.xyz --genes GENES.txt --output runs/hbond_double \
  --extra-objectives 'hbond_='
```

Double bonds are detected once from the reference XYZ. The score is continuous
and is maximized for a donor-specific H-to-center distance, a linear
X-H...center contact, and an approach close to the perpendicular bisector of
the bond. `hbond_double` and `hbond_eq` are accepted aliases.

The live TUI is enabled with `--tui`. Its active controls are shown in the
header: `[m]` changes overview/gallery mode, `[1-9]` focuses an island, and
`[n]`/`[p]` move between focused islands. Set `NO_COLOR=1`, use `--no-tui`, or
redirect output for non-interactive environments. Set `SEEKER_ANIMATION=0`
to disable startup animation.

## 4. Outputs and analysis

The run directory contains the manifest, checkpoint, energy cache, evaluated
archive, final population, Pareto front, selected candidates, and analysis
reports. These are reproducible products and must not be committed.

Re-run analysis with different clustering parameters:

```bash
seeker analyze --run runs/MY_RUN --source archive \
  --method hybrid --max-delta-energy 10 \
  --torsion-mean-threshold-deg 15 --torsion-max-threshold-deg 15
```

For `pose_hybrid`, exchanges of graph-isomorphic repeated fragments are enabled
by default. Historical label-sensitive selection can be reproduced with
`--pose-permutation-mode ordered`.

Optimize selected candidates with xTB:

```bash
seeker optimize --input-dir runs/MY_RUN/selected_candidates \
  --output-dir runs/MY_RUN/postoptimization \
  --xtb-method gfn2 --jobs 4 --opt-level tight
```

When `--xtb-command` is omitted, this command uses the xTB executable saved by
`seeker configure`, including a machine-local patched build.

Apply the fixed `B3LYP/6-31+G* EmpiricalDispersion=GD3BJ` filter before
optimization:

```bash
seeker single-point \
  --input-dir runs/MY_RUN/selected_candidates \
  --output-dir runs/MY_RUN/postoptimization/single_points \
  --energy-window 10
```

Optimize the survivors with B3LYP:

```bash
seeker optimize --backend gaussian \
  --input-dir runs/MY_RUN/postoptimization/single_points/filtered_xyz \
  --output-dir runs/MY_RUN/postoptimization \
  --constraint-mode active \
  --run-manifest runs/MY_RUN/run_manifest.json
```

`active` freezes intramolecular bonds, angles, and non-genetic torsions while
leaving genetic `D(...)` bonds and relative `POSE` motion free. It is exact
only with Gaussian and is rejected for runs containing `RING(...)`. Gaussian
jobs, `%nprocshared`, and `%mem` use conservative machine-derived defaults and
can be overridden with `--jobs`, `--gaussian-nprocshared`, and
`--gaussian-mem-gb`.

Final optimized structures can be clustered with:

```bash
seeker cluster-optimized \
  --optimization-csv runs/MY_RUN/postoptimization/optimization.csv \
  --output-dir runs/MY_RUN/postoptimization/clustering \
  --rmsd-threshold 0.30 --energy-window 10 --atom-mode all \
  --permutation-mode equivalent
```

`equivalent` is the default and exchanges only graph-isomorphic disconnected
fragments and graph-equivalent atoms. It therefore removes duplicate solvent
labels without merging different inferred topologies. Use
`--permutation-mode ordered` to reproduce strict historical XYZ-index RMSD.

## 5. Optional installations

```bash
python -m pip install -e '.[discovery]'  # HDBSCAN/hybrid analysis
python -m pip install -e '.[pyscf]'      # PySCF energy backend
```

The repository does not bundle xTB, Open Babel, Gaussian, CREST, or any
patched executable.

## 6. Troubleshooting

- Run `seeker doctor` first when a backend is missing.
- Use a new output directory, or pass an explicit checkpoint with `--resume`.
- Use `--validate-only` to isolate input/coordinate failures from energy-backend
  failures.
- Hybrid analysis requires the `discovery` extra.
