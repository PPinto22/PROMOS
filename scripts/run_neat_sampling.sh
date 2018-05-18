#!/bin/bash
evolver="../src/py/evolver.py"
options="-d ../data/data_train.csv -t ../data/data_test.csv -o ../results/NEAT/samples -m neat -T 30 -p 54"

# Do 30 runs
for i in {1..30} 
do
	python ${evolver} ${options} -s 100 --id="100(${i})"
	python ${evolver} ${options} -s 1000 --id="1K(${i})"
	python ${evolver} ${options} -s 10000 --id="10K(${i})"
	python ${evolver} ${options} --id="ALL(${i})"
done

exit 0