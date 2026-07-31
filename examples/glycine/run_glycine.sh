#!/usr/bin/env zsh
set -euo pipefail
setopt PIPE_FAIL
unsetopt BG_NICE

SCRIPT_DIR="${0:A:h}"
SCRIPT_NAME="${0:t}"
SEEKER_ROOT="${SCRIPT_DIR:h:h}"
LAUNCHER="$SEEKER_ROOT/run_seeker.sh"
SAVE_GENERATION_XYZ=0
OUTPUT_DIR=""

usage() {
  print -r -- "Uso: $SCRIPT_NAME [--save-generation-xyz] [DIRECTORY_OUTPUT]"
  print -r -- ""
  print -r -- "Esegue la configurazione validata della glicina senza domande:"
  print -r -- "SCAN 3×2×2, seed 7, popolazione 12, 24 figli, 10 generazioni,"
  print -r -- "GFN2-xTB e clustering ibrido entro 15 kcal/mol (massimo 16 candidati)."
}

for argument in "$@"; do
  case "$argument" in
    -h|--help)
      usage
      exit 0
      ;;
    --save-generation-xyz)
      SAVE_GENERATION_XYZ=1
      ;;
    --*)
      print -u2 -r -- "Opzione non riconosciuta: $argument"
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$OUTPUT_DIR" ]]; then
        print -u2 -r -- "Indicare al massimo una directory di output."
        usage >&2
        exit 2
      fi
      OUTPUT_DIR="${argument:A}"
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$SCRIPT_DIR/output/glycine_seed7_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  print -u2 -r -- "La directory di output esiste già: $OUTPUT_DIR"
  exit 2
fi
if [[ ! -x "$LAUNCHER" ]]; then
  print -u2 -r -- "Launcher SEEKER non trovato o non eseguibile: $LAUNCHER"
  exit 2
fi

# Configurazione scientifica congelata della run validata.
export PYTHONDONTWRITEBYTECODE=1
export CHARGE=0
export MULTIPLICITY=1
export INITIAL_POPULATION=scan
export SCAN_LAYOUT=tensor
export INITIAL_SCAN_GRID=periodicity-modes
export SCAN_POINTS_MODE=periodicity
export SCAN_POINTS=3
export POPULATION=12
export OFFSPRING=24
export GENERATIONS=10
export SEED=7
export BATCH_WORKERS=4
export BASE_MUTATION_WEIGHT=0.45
export BASE_CROSSOVER_WEIGHT=0.80
export MUTATION_WEIGHT_AMPLITUDE=0.10
export CROSSOVER_WEIGHT_AMPLITUDE=0.10
export OPERATOR_OSCILLATIONS=2
export PERIODICITY_GRID_STEP=20.0
export SBX_ETA=15.0
export DUPLICATE_MEAN_THRESHOLD=3.0
export DUPLICATE_MAX_THRESHOLD=3.0
export MAX_DUPLICATE_ATTEMPTS=30
export GEOMETRIC_PRESCREEN=1
export TOPOLOGY_TOLERANCE=0.45
export HBOND_CUTOFF=3.2
export HBOND_CONTACT_THRESHOLD=-0.30
export HH_CLASH_DISTANCE=1.40
export STERIC_HH_SCALE=0.55
export STERIC_HEAVY_HEAVY_SCALE=0.55
export STERIC_HYDROGEN_HEAVY_SCALE=0.50
export STERIC_EXCLUDE_HOPS=3
export EXTRA_OBJECTIVES=""
export CHARGE_MODEL=""
export EARLY_STOP=0
export ARCHIVE_STAGNATION_PATIENCE=0
export CLUSTERING_SOURCE=archive
export CLUSTERING_METHOD=hybrid
export CLUSTER_MEAN_THRESHOLD=15.0
export CLUSTER_MAX_THRESHOLD=15.0
export CLUSTER_ENERGY_WINDOW=15.0
export HYBRID_MAX_CANDIDATES=16
export HYBRID_MIN_CLUSTER_SIZE=5
export HYBRID_MIN_SAMPLES=2
export HYBRID_ENERGY_NEIGHBORS=8
export HYBRID_MIN_SEPARATION=25.0
export SAVE_GENERATION_XYZ
export ENERGY_OWNER=driver
export SEEKER_BACKEND=xtb
export ENERGY_TIMEOUT=600.0
export XTB_COMMAND="${XTB_COMMAND:-xtb}"
export XTB_METHOD=gfn2
export XTB_THREADS=1

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run.log"

print -r -- "SEEKER · configurazione glicina validata"
print -r -- "Input:  $SCRIPT_DIR/input.xyz"
print -r -- "Genes:  $SCRIPT_DIR/genes.txt · periodicità (3,2,2)"
print -r -- "Output: $OUTPUT_DIR"
print -r -- "XYZ per generazione: $([[ "$SAVE_GENERATION_XYZ" == "1" ]] && print si || print no)"
print -r -- ""

"$LAUNCHER" --worker \
  "$SCRIPT_DIR/input.xyz" \
  "$SCRIPT_DIR/genes.txt" \
  "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"

print -r -- ""
print -r -- "Run glicina completata."
print -r -- "Candidati conformazionali: $OUTPUT_DIR/selected_candidates/"
print -r -- "Evoluzione genetica:       $OUTPUT_DIR/genetic_evolution.csv"
print -r -- "Clustering:                 $OUTPUT_DIR/analysis/summary.json"
