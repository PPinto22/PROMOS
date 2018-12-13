#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best_idf_train.csv -t ../data/2weeks/best_idf_test.csv -P ../params/neat.txt -o ../results/neat_limited_best_idf -b ../cfg/bloat_mut_con.cfg -m neat -g10000 -s2000 -p6 --no-statistics --quiet"
runs=4
parallel=1

function run() {
	children=$(pgrep -c -P$$)
	if (( $children >= $parallel )); then
		echo "Waiting: ${children} tasks running..."
		wait -n;
	fi
	echo "Starting: $evolver $options $1"
	( $evolver $options $1 ) &
}

for i in $(seq 1 $runs)	
do
	run "--id=best_idf(${i})"
done
wait
exit 0
