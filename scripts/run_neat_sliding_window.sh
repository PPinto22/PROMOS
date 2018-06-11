#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/week1_best/all.csv"
options="-P ../params/neat0.txt -o ../results/window -m neat -g750 -s1000 -p7 -W96 -w24 -S24 --test-fitness --quiet"
runs=16
parallel=8

for i in $(seq 1 $runs)
do
	echo "Run $i..."
	( $evolver $args $options -s 100 --id="windows(${i})" ) &
	if (( $i % $parallel == 0 )); then wait; fi
done
wait
exit 0
