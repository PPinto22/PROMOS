#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/test_raw_train.csv -t ../data/2weeks/test_raw_test.csv -P ../params/neat.txt -o ../results/neat_test_raw_plot -m neat -g10000 -s2000 -p6 --test-fitness --quiet"
runs=4
parallel=4

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
