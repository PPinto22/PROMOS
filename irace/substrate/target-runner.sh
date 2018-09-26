#!/bin/bash
CONFIG_ID=$1
INSTANCE_ID=$2
SEED=$3
INSTANCE=$4
TRAIN_DATA=$5
TEST_DATA=$6
PARAMS=$7
shift 7 || exit 1
CONFIG_PARAMS=$*

EXE="python ../../src/py/evolver.py"
FIXED_PARAMS="-d ${TRAIN_DATA} -t ${TEST_DATA} -P ${PARAMS} -s2000 -T8 -p6 -o NULL --quiet --no-reevaluation"

STDOUT=c${CONFIG_ID}-${INSTANCE_ID}.stdout
STDERR=c${CONFIG_ID}-${INSTANCE_ID}.stderr

error() {
  echo "`TZ=UTC date`: error: $@"
  exit 1
}

if [ ! "$(command -v ${EXE})" ]; then
    error "$EXE: not found or not executable (pwd: $(pwd))"
fi

{ $EXE ${FIXED_PARAMS} --seed ${SEED} ${CONFIG_PARAMS}; } 1> ${STDOUT} 2> ${STDERR}

if [ -s "${STDOUT}" ]; then
    value=-$(sed -rz 's/.*Fitness \(test\): ([+-]?([0-9]*[.])?[0-9]+).*/\1/' ${STDOUT})
    if ! [[ $value =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
      error "Return value \"${value}\" is not a valid number"
      exit 1
    fi
    echo $value
else
    error "${STDOUT}: No such file or directory"
fi

# Clean up
# rm -f "${STDOUT}" "${STDERR}" "${PARAM_FILE}"
exit 0
