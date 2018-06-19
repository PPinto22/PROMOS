#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best.csv"
options="-P ../params/irace2_balanced.txt -o ../results/2weeks_balanced/ -m neat -g750 -s1000 -p13 -W120 -w24 -S24 --test-fitness"
runs=8
parallel=4

for i in $(seq 1 $runs)
do
	echo "Run $i..."
	( $evolver $args $options --id="windows(${i})" ) &
	if (( $i % $parallel == 0 )); then wait; fi
done
wait
exit 0
