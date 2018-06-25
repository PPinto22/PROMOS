#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best.csv"
options="-P ../params/irace2_zero.txt -o ../results/zero/ -m neat -g200 -s1000 -p4 -W120 -w24 -S24 --test-fitness"
runs=1
parallel=1

for i in $(seq 1 $runs)
do
	echo "Run $i..."
	( $evolver $args $options --id="zero" ) &
	if (( $i % $parallel == 0 )); then wait; fi
done
wait
exit 0
