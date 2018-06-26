#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best.csv"
options="-P ../params/irace2.txt -o ../results/2weeks_bloat_mut_con/ -m neat -g750 -s1000 -p13 -W120 -w24 -S24 --test-fitness -b ../params/bloat_mut_con.cfg"
runs=4
parallel=4

for i in $(seq 1 $runs)
do
	echo "Run $i..."
	( $evolver $args $options --id="bloat_mut_con(${i})" ) &
	if (( $i % $parallel == 0 )); then wait; fi
done
wait
exit 0
