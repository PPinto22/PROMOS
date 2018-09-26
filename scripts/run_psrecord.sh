#!/bin/bash

clean_up() {
	kill $EVOLVER_PID
	kill $PSRECORD_PID
	exit 1
}

trap clean_up SIGHUP SIGINT SIGTERM

python ../src/py/evolver.py -d ../data/2weeks/best_idf.csv -P ../params/neat.txt -o ../results/psrecord/ -m neat -g1500 -s1000 -p10 --no-reevaluation --id="psrecord" &
EVOLVER_PID=$!

psrecord $! --interval 1 --log ../logs/psrecord.txt --plot ../logs/psrecord.png --include-children &
PSRECORD_PID=$!

wait
exit 0