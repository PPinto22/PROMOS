#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ../data/2weeks/best_idf_train.csv -P ../params/neat.txt -o ../results/processes -m neat -g100 --no-statistics"
runs=1

for i in $(seq 1 $runs)
do
	for p in {1,2,4,8,16,30,40,50} 
	do
		echo "Processes: $p | Run: $i..."
		( $evolver $options -p$p --id="$p($i)" )
	done
done
wait
exit 0
