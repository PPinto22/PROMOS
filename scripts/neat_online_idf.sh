#!/bin/bash
python ../src/py/evolver.py -P ../params/neat_online.txt -E ../cfg/encoder_idf.cfg -b ../cfg/bloat_online_idf.cfg -o ../results/online_idf -m neat -T14400 -s2000 -p6 --test-fitness -S6 -W96 -w0.3 --online --id=online $@
