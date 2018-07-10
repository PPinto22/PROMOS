#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best_idf.csv"
id="balanced_mut"
options="-P ../params/irace2_balanced.txt -o ../results/${id}/ -m neat -g750 -s2500 -p13 -W120 -w24 -S24 --test-fitness"
runs=8
parallel=4

for i in $(seq 1 $runs)
do
	echo "Run $i..."
	( $evolver $args $options --id="${id}(${i})" ) &
	if (( $i % $parallel == 0 )); then wait; fi
done
wait
exit 0
