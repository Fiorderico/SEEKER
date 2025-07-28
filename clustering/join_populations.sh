#!/bin/bash

# Verifica che siano stati passati due argomenti (i e j)
if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <valore_iniziale> <valore_finale>"
    exit 1
fi

start=$1
end=$2

# Crea la cartella cumulative_population se non esiste
if [ ! -d "cumulative_population" ]; then
    mkdir cumulative_population
fi

# Itera dalle cartelle population_start a population_end
for (( i=start; i<=end; i++ ))
do
    dir="../generations_gly/population_$i"
    if [ -d "$dir" ]; then
        # Copia i file .xyz presenti nella cartella corrente
        cp "$dir"/*.xyz cumulative_population/ 2>/dev/null
    else
        echo "La cartella $dir non esiste."
    fi
done

echo "Copia completata."

