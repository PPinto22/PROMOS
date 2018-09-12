#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best_pcp_train.csv -t ../data/2weeks/best_pcp_test.csv -P ../params/neat.txt -o ../results/2wks_best_pcp -m neat -g10000 -s2000 -p6 --test-fitness --quiet"
runs=5
parallel=5

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
	run "--id=best_pcp(${i})"
done
wait
exit 0
