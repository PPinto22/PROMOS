#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best_pcp_train.csv -t ../data/2weeks/best_pcp_test.csv -P ../params/hyperneat.txt -o ../results/2wks_hn_best_pcp -m hyperneat -g1000 -s2000 -p6 --test-fitness --quiet"
runs=3
parallel=3

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
