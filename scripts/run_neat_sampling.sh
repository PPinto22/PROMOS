#!/bin/bash
evolver="python ../src/py/evolver.py"
args="../data/data_train.csv"
options="-t ../data/data_test.csv -o ../results/TEMP -m neat -T 30 -p 54"

# Do 30 runs
for i in {1..30} 
do
	${evolver} ${args} ${options} -s 100 --id="100(${i})"
	${evolver} ${args} ${options} -s 1000 --id="1K(${i})"
	${evolver} ${args} ${options} -s 10000 --id="10K(${i})"
	${evolver} ${args} ${options} --id="ALL(${i})"
done

exit 0
