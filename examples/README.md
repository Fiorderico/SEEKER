# Source examples

This directory contains small, reproducible inputs only. Generated runs belong
under the repository-level `runs/` directory and are ignored by Git.

| Example | Purpose |
|---|---|
| `glycine/` | Fast torsion and CLI validation |
| `erythrulose/` | Intramolecular hydrogen-bond search |
| `fructose/` | Flexible torsional input |
| `thiopronine/` | Larger acyclic torsional input |
| `intermoleculars/imidazole_h2o/` | Native rigid-fragment POSE |

Validate any example without an energy calculation:

```bash
seeker run --xyz examples/erythrulose/input.xyz \
  --genes examples/erythrulose/genes.txt \
  --output runs/erythrulose_validation --validate-only
```

Reference conformer collections, matching matrices, optimized structures, and
previous run archives are deliberately not distributed in this repository.
