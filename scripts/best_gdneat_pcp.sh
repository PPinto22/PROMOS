#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best_pcp_train.csv -t ../data/2weeks/best_pcp_test.csv -P ../params/neat.txt -o ../results/gdneat_pcp_best -m gdneat -g5000 -s2000 -p4 --test-fitness"
runs=1
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
	run "--id=gdneat_pcp_best(${i})"
done
wait
exit 0
