#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best.csv -P ../params/neat.txt -o ../results/2wks_sw_best_pcp_05 -E ../cfg/encoder_pcp_05.cfg -m neat -g1000 -s2000 -p6 --no-statistics -W120 -w24 -S24"
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
	run "--id=sw_best_pcp(${i})"
done
wait
exit 0
