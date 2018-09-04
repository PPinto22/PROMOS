#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best_idf_train.csv"
options="-t ../data/2weeks/best_idf_val.csv -P ../params/neat.txt -o ../results/2wks_mut_con -m neat -g7500 -s2000 -p6 --test-fitness --quiet"
runs=10 # per run type
parallel=5

function run() {
	children=$(pgrep -c -P$$)
	if (( $children >= $parallel )); then
		echo "Waiting: ${children} tasks running..."
		wait -n;
	fi
	echo "Starting: $evolver $args $options $1"
	( $evolver $args $options $1 ) &
}

for i in $(seq 1 $runs)	
do
	run "-b ../cfg/bloat_mut_con.cfg --id=mut(${i})"
done
wait
exit 0
