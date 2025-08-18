# Example Gaussian Inputs

Sample `.gjf` files for quick testing of different molecules.

- `butano.gjf`
- `gly.gjf`
- `thiopronine_g16.gjf`
- `thiopronine_g16_to_use.gjf`
- `input_reference.gjf`

Use these as templates or references when configuring `INPUT_FILE` in `main.py`.

## Gene Definitions

Below are example `GENI` lists to pair with the inputs:

### Glycine
```python
GENI = [
    (3, "D(  6,  1,  2,  3)"),
    (3, "D(  1,  2,  3,  4)"),
    (2, "D(  4,  3,  5, 10)")
]
```

### Butano
```python
GENI = [
    (3, "D(5,1,2,3)"),
    (3, "D(1,2,3,4)"),
    (3, "D(2,3,4,13)")
]
```

The `GENI` definition for tiopronin is already present in `main.py`.
