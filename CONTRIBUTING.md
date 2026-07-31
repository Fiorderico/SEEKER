# Contributing to SEEKER

SEEKER supports Python 3.10 or newer on macOS and Linux.

```bash
python3 --version  # must report 3.10 or newer
python3 -c 'import sys; assert sys.version_info >= (3,10), sys.version'
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[discovery]'
seeker --version
seeker doctor
seeker run --xyz examples/glycine/input.xyz \
  --genes examples/glycine/genes.txt \
  --output runs/development-validation --validate-only --no-tui
```

Keep source inputs, scripts, and compact synthetic fixtures under version
control. Do not commit run directories, electronic-structure logs, generated
XYZ archives, post-optimization results, or local executable paths. Use
`git check-ignore -v PATH` before adding a new scientific artifact.

Internal tests, benchmarks, temporary files, caches, and scientific results are
not distributed in the public-facing repository. Before contributing, verify a
fresh editable installation and the validation smoke test above.

This repository currently has no open-source license. Opening an issue or pull
request does not change the copyright status of the existing code.
