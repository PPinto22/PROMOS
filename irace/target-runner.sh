#!/bin/bash
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# $1 is the candidate configuration number
# $2 is the instance ID
# $3 is the seed
# $4 is the instance name
# The rest ($* after `shift 4') are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################
CONFIG_ID=$1
INSTANCE_ID=$2
SEED=$3
INSTANCE=$4
TRAIN_DATA=$5
TEST_DATA=$6
shift 6 || exit 1
CONFIG_PARAMS=$*

EXE="python ../../src/py/evolver.py"
PARAM_WRITER=../param-writer.sh
FIXED_PARAMS="${TRAIN_DATA} -t ${TEST_DATA} -T10 -p7 -s1000 -o NULL --quiet"

STDOUT=c${CONFIG_ID}-${INSTANCE_ID}.stdout
STDERR=c${CONFIG_ID}-${INSTANCE_ID}.stderr
PARAM_FILE=c${CONFIG_ID}-${INSTANCE_ID}.params.txt

error() {
  echo "`TZ=UTC date`: error: $@"
  exit 1
}

for cmd in "${EXE}" "${PARAM_WRITER}"
do
  if [ ! "$(command -v $cmd)" ]; then
      error "$cmd: not found or not executable (pwd: $(pwd))"
  fi
done

# Call PARAM_WRITER to write parameters to PARAM_FILE
($PARAM_WRITER ${CONFIG_PARAMS}) 1> ${PARAM_FILE}

# If the program just prints a number, we can use 'exec' to avoid
# creating another process, but there can be no other commands after exec.
#exec $EXE ${FIXED_PARAMS} -i $INSTANCE ${CONFIG_PARAMS}
# exit 1
#
# Otherwise, save the output to a file, and parse the result from it.
# (If you wish to ignore segmentation faults you can use '{}' around
# the command.)
{ $EXE ${FIXED_PARAMS} -P ${PARAM_FILE} --seed ${SEED}; } 1> ${STDOUT} 2> ${STDERR}

# # This may be used to introduce a delay if there are filesystem
# # issues.
# SLEEPTIME=1
# while [ ! -s "${STDOUT}" ]; do
#     sleep $SLEEPTIME
#     let "SLEEPTIME += 1"
# done

# This is an example of reading a number from the output.
# It assumes that the objective value is the first number in
# the first column of the last line of the output.
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
rm -f "${STDOUT}" "${STDERR}" "${PARAM_FILE}"
exit 0
