#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks_idf/best.csv"
id="2weeks_idf"
options="-P ../params/irace2.txt -o ../results/${id}/ -m neat -g750 -s2500 -p13 -W120 -w24 -S24 --test-fitness"
runs=2
parallel=2

for i in $(seq 1 $runs)
do
	echo "Run $i..."
	( $evolver $args $options --id="${id}(${i})" ) &
	if (( $i % $parallel == 0 )); then wait; fi
done
wait
exit 0
