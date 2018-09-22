#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/test_pcp_train.csv -t ../data/2weeks/test_pcp_test.csv -P ../params/hyperneat.txt -o ../results/2wks_hn_test_pcp -m hyperneat -g10000 -s2000 -p6 --test-fitness --substrate-width=4 --substrate-length=8"
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
	run "--id=test_pcp(${i})"
done
wait
exit 0
