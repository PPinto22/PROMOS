#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best_idf.csv"
id="sw_2wks_idf_step_long"
options="-P ../params/neat.txt -o ../results/${id}/ -m neat -g1500 -s2500 -p13 -W120 -w24 -S24 --test-fitness -b ../cfg/bloat_fit_step.cfg"
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
