#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best.csv"
id="sw_2wks_idf"
options="-P ../params/neat.txt -o ../results/${id}/ -E ../cfg/encoder_idf.cfg -m neat -g750 -s2500 -p10 -W120 -w24 -S24 --test-fitness"
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
