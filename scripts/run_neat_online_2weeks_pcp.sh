#!/bin/bash
python ../src/py/evolver.py -P ../params/neat.txt -E ../cfg/encoder_pcp.cfg -o ../results/online_pcp -m neat -T20160 -s2000 -p6 --test-fitness -S6 -W96 -w0.3 --online --id=online