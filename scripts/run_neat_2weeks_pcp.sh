#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best_pcp_train.csv"
id="2wks_pcp"
options="-t ../data/2weeks/best_pcp_val.csv -P ../params/neat.txt -o ../results/${id}/ -m neat -g5000 -s2500 -p10 --test-fitness"
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
