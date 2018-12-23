#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/test.csv -P ../params/neat.txt -o ../results/test_sw_neatp_raw -E ../cfg/encoder_raw.cfg -m neat -b ../cfg/bloat_mut_con.cfg -g1000 -s2000 -p6 --no-statistics -W120 -w24 -S24 --quiet"
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
