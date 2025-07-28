# Genetic Optimization of Molecular Geometries

This project implements a **genetic algorithm** for the optimization of molecular geometries through **torsional degrees of freedom**. It generates modified Gaussian input files (`.gjf`), evaluates them with an external software (`gdv`), and extracts optimized geometries in `.xyz` format.

## Overview

Given a reference `.gjf` input file, the code applies torsional modifications (genes) to specific dihedral angles and evolves a population of candidate structures to **minimize the Hartree–Fock energy** (HF). Each generation retains the best individuals and produces new ones via **crossover and mutation**.  Calculations are executed by the external program `gdv` whose output is parsed to obtain the fitness value.

## Algorithm Details

An individual is defined by the list of dihedral angles specified in `GENI`.  For each gene with periodicity `n` a random initial allele θ in degrees is generated from the importance-sampling probability

\[
P(\theta) = b + w\bigl(1 + \cos(n\theta)\bigr),\qquad
b = \frac{1}{2\pi} - w,
\]

with `w=0.08`.  Candidate values are sampled uniformly in radians and accepted with probability

\[
\frac{b + w\bigl(1 + \cos(n\theta)\bigr)}{\tfrac{1}{2\pi}+w}.
\]

At each generation `g` the mutation and crossover probabilities oscillate according to

\[
r_{\text{mut}}(g) = r_{0}^{\text{mut}} + \Delta r_{\text{mut}}\sin\Bigl(\frac{\pi\,n_{\text{osc}}}{N-1}g\Bigr),\\[2mm]
r_{\text{cross}}(g) = r_{0}^{\text{cross}} - \Delta r_{\text{cross}}\sin\Bigl(\frac{\pi\,n_{\text{osc}}}{N-1}g\Bigr),
\]

where `N` is the total number of generations, `n_osc` is `NUM_OSCILLATIONS` and `r_0` and Δ are the base values and amplitudes defined in `main.py`.

Evaluation consists of:

1. removing the `frozen` keyword from the reference `.gjf` lines,
2. inserting the `GENE` definitions with the current alleles,
3. running `gdv` to obtain the log file,
4. extracting the last `HF=` value as fitness and the optimized coordinates to produce a `.xyz` file.

The population is sorted by increasing HF energy and the best `POPOLAZIONE_TARGET` individuals are kept. Parents are chosen via **tournament selection** and children inherit each allele from a random parent (**uniform crossover**).  With probability `r_{mut}` a gene is replaced by a newly sampled angle.

An optional similarity penalty (currently commented in `main.py`) can be enabled to discourage individuals that are too close in allele space.  Similarity between two vectors `A` and `B` is computed as

\[
S(A,B) = \exp\bigl(-\gamma\|A-B\|^{2}\bigr)
\]

and a penalty is applied if `S(A,B)` exceeds the threshold `SIMILARITY_THRESHOLD`.

## Project Structure

```
.
├── input_reference.gjf         # Reference Gaussian input file (modifiable)
├── main.py                     # Main script with genetic algorithm
├── <GENERATIONS_DIR>/          # Output directory set by `GENERATIONS_DIR`
│   └── population_*/           # Contains `.xyz` files of evaluated structures
├── tmp/                        # Temporary working directory
└── README.md                   # This file
```

## Configuration

Inside `main.py`, you can configure:

* `INPUT_FILE`: reference .gjf file to mutate
* `GENI`: list of dihedral definitions with periodicity
* `NUM_GENERAZIONI`: number of generations
* `POPOLAZIONE_INIZIALE` and `POPOLAZIONE_TARGET`: initial and retained population sizes
* `BASE_RATE_MUTATION` and `BASE_RATE_CROSSOVER`: base probabilities
* `DELTA_RATE_MUTATION` and `DELTA_RATE_CROSSOVER`: oscillation amplitudes
* `NUM_OSCILLATIONS`: number of oscillatory cycles along the run

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

4. Optimized `.xyz` geometries and energy logs will be stored inside `GENERATIONS_DIR` in subfolders `population_<gen>/`.

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
