# Imidazole–water rigid-fragment pose

`GENES.txt` keeps imidazole as the reference fragment and moves water rigidly.
The centroid distance spans 2.6–5.2 Å, the direction covers the full sphere,
and the relative orientation is bounded to 118° from `input.xyz`.

Validate the native pose definition without energy evaluation:

```bash
seeker run \
  --xyz examples/intermoleculars/imidazole_h2o/input.xyz \
  --genes examples/intermoleculars/imidazole_h2o/GENES.txt \
  --output runs/imidazole_h2o_validation \
  --validate-only
```

Validation checks the disconnected-fragment topology, centroid displacement,
relative rotation, and fragment rigidity before the genetic search starts.
