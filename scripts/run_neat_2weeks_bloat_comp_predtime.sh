#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best_idf_train.csv -t ../data/2weeks/best_idf_val.csv -P ../params/neat.txt -o ../results/2wks_bloat_comp -m neat -g7500 -s2000 -p6 --test-fitness --quiet"
runs=6 # per run type
parallel=8

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
	run "--id=none(${i})"
	run "-b ../cfg/bloat_fit_step_predtime.cfg --id=step(${i})"
	run "-b ../cfg/bloat_fit_popmax_predtime.cfg --id=popmax(${i})"
	run "-b ../cfg/bloat_mut_predtime.cfg --id=mut(${i})"
done
wait
exit 0
