#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/2weeks/best_idf_train.csv"
options="-P ../params/irace2.txt -o ../results/processes -m neat -g100 -s2500 --no-statistics"
runs=5

for i in $(seq 1 $runs)
do
	for p in {1,2,4,8,16,32,40,50,60} 
	do
		echo "Processes: $p | Run: $i..."
		( $evolver $args $options -p$p --id="$p($i)" )
	done
done
wait
exit 0
