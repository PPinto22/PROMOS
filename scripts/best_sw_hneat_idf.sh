#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best.csv -P ../params/hyperneat.txt -o ../results/best_sw_hneat_idf -E ../cfg/encoder_idf.cfg -m hyperneat -g1000 -s2000 -p6 --no-statistics -W120 -w24 -S24 --quiet"
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
