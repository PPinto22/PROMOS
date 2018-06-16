#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best.csv"
options="-P ../params/irace2.txt -o ../results/2weeks/ -m neat -g750 -s1000 -p7 -W120 -w24 -S24 --test-fitness --quiet"
runs=8
parallel=8

for i in $(seq 1 $runs)
do
	echo "Run $i..."
	( $evolver $args $options -s 100 --id="windows(${i})" ) &
	if (( $i % $parallel == 0 )); then wait; fi
done
wait
exit 0
