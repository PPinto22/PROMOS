#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best_raw_train.csv -t ../data/2weeks/best_raw_test.csv -P ../params/hyperneat.txt -o ../results/2wks_hn_best_raw -m hyperneat -g10000 -s2000 -p6 --no-statistics --quiet"
runs=2
parallel=2

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
	run "--id=best_raw(${i})"
done
wait
exit 0
