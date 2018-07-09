#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks_idf/best.csv"
id="bloat_fit_step_long"
options="-P ../params/irace2.txt -o ../results/${id}/ -m neat -g1500 -s2500 -p13 -W120 -w24 -S24 --test-fitness -b ../params/bloat_fit_step.cfg"
runs=4
parallel=4

for i in $(seq 1 $runs)
do
	echo "Run $i..."
	( $evolver $args $options --id="${id}(${i})" ) &
	if (( $i % $parallel == 0 )); then wait; fi
done
wait
exit 0
