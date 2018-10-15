#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/test.csv -P ../params/hyperneat.txt -o ../results/2wks_hn_sw_test_pcp -E ../cfg/encoder_pcp.cfg -m hyperneat -g1000 -s2000 -p6 --test-fitness -W120 -w24 -S24 --substrate-width=4 --substrate-length=8"
runs=2
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
	run "--id=sw_hn_test_pcp(${i})"
done
wait
exit 0
