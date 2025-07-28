# Genetic Optimization of Molecular Geometries

This project implements a **genetic algorithm** for the optimization of molecular geometries through **torsional degrees of freedom**. It generates modified Gaussian input files (`.gjf`), evaluates them with an external software (`gdv`), and extracts optimized geometries in `.xyz` format.

## Overview

Given a reference `.gjf` input file, the code applies torsional modifications (genes) to specific dihedral angles and evolves a population of candidate structures to **minimize the Hartree–Fock energy** (HF). Each generation retains the best individuals and produces new ones via **crossover and mutation**.

## Project Structure

```
.
├── input_reference.gjf         # Reference Gaussian input file (modifiable)
├── main.py                     # Main script with genetic algorithm
├── generations_gly_prob/       # Output directory with one subfolder per generation
│   └── population_*/           # Contains .xyz files of evaluated structures
├── tmp/                        # Temporary working directory
└── README.md                   # This file
```

## Configuration

Inside `main.py`, you can configure:

* `INPUT_FILE`: reference .gjf file to mutate
* `GENI`: list of dihedral definitions with periodicity
* `NUM_GENERAZIONI`: number of generations
* `POPOLAZIONE_INIZIALE` and `POPOLAZIONE_TARGET`: initial and retained population sizes
* `MUTATION_RATE` and `CROSSOVER_RATE`: evolution parameters

Example of `GENI`:

```python
GENI = [
    (3, "D(  6,  1,  2,  3)"),
    (3, "D(  1,  2,  3,  4)"),
    (2, "D(  4,  3,  5, 10)")
]
```

## How to Run

1. Make sure the binary `gdv` is available in your `PATH`. This should accept `.gjf` from stdin and print output to stdout.
2. Place your reference input file (e.g., `input_reference.gjf`) in the root directory.
3. Run the script:

```
python main.py
```

4. Optimized `.xyz` geometries and energy logs will be stored in the `generations_gly_prob/` directory.

## Output

Each generation will output:

* `.xyz` files with HF energy in the second line
* `generation_log.txt` with average, min, and max fitness per generation

Example `.xyz` file:

```
12
HF=-154.234567
6 -0.123 1.234 0.456
1  0.456 0.789 -0.987
...
```

## Notes

* The genetic algorithm uses importance sampling for initial allele generation based on periodic cosine-based distributions.
* You can easily modify the number and definition of genes to adapt to different molecules.
* The `frozen` keyword in the reference `.gjf` is automatically removed when preparing inputs.

## License

MIT License
