#!/bin/bash
evolver="python ../src/py/evolver.py"
options="-d ./data/2weeks/best_idf_train.csv -t ../data/2weeks/best_idf_test.csv -P ../params/neat.txt -o ../results/sampling -m neat -T20 -p8"
runs=10

for i in $(seq 1 $runs)
do
	( ${evolver} ${options} -s 100 --id="100(${i})" ) &
	( ${evolver} ${options} -s 1000 --id="1K(${i})" ) &
	( ${evolver} ${options} -s 10000 --id="10K(${i})" ) &
	( ${evolver} ${options} --id="ALL(${i})" ) &
	wait;
done
exit 0
