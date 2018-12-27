#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best_raw_train.csv -t ../data/2weeks/best_raw_test.csv -P ../params/hyperneat.txt -o ../results/best_hn_raw -m hyperneat -g10000 -s2000 -p6 --quiet"
runs=9
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
	run "--id=run(${i})"
done
wait
exit 0
