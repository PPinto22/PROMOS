#!/bin/bash
python ../src/py/evolver.py -P ../params/neat_online.txt -E ../cfg/encoder_pcp.cfg -b ../cfg/bloat_online.cfg -o ../results/online_pcp -m neat -T14400 -s2000 -p6 --test-fitness -S6 -W96 -w0.3 --online --id=online $@
