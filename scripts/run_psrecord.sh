#!/bin/bash

clean_up() {
	kill $EVOLVER_PID
	kill $PSRECORD_PID
	exit 1
}

trap clean_up SIGHUP SIGINT SIGTERM

python ../src/py/evolver.py ../data/2weeks/best.csv -P ../params/irace2.txt -o ../results/psrecord/ -m neat -g750 -s1000 -p4 -W120 -w24 -S24 --test-fitness --id="psrecord" &
EVOLVER_PID=$!

psrecord $! --interval 1 --log ../logs/psrecord.txt --plot ../logs/psrecord.png &
PSRECORD_PID=$!

wait
exit 0