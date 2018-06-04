#!/bin/bash

# Input example: PopulationSize=100 MutationRate=0.05
arguments=$*

activation_functions() {
  usigmoid=0; ugauss=0; usine=0; relu=0
  case $1 in
    0)  usigmoid=1;;
    1)  ugauss=1;;
    2)  usine=1;;
    3)  relu=1;;
    4)  usigmoid=0.5; ugauss=0.5;;
    5)  usigmoid=0.5; usine=0.5;;
    6)  usigmoid=0.5; relu=0.5;;
    7)  ugauss=0.5; usine=0.5;;
    8)  ugauss=0.5; relu=0.5;;
    9)  usine=0.5; relu=0.5;;
    10) usigmoid=0.334; ugauss=0.333; usine=0.333;;
    11) usigmoid=0.334; ugauss=0.333; relu=0.333;;
    12) usigmoid=0.334; usine=0.333; relu=0.333;;
    13) ugauss=0.334; usine=0.333; relu=0.333;;
    14) usigmoid=0.25; ugauss=0.25; usine=0.25; relu=0.25;;
  esac
  echo "ActivationFunction_UnsignedSigmoid_Prob ${usigmoid}"
  echo "ActivationFunction_UnsignedGauss_Prob ${ugauss}"
  echo "ActivationFunction_UnsignedSine_Prob ${usine}"
  echo "ActivationFunction_Relu_Prob ${relu}"
}

# Read parameters into array
IFS=' ' read -r -a params <<< $arguments

# Begin tag
echo -e "NEAT_ParametersStart\n"

# Static parameters
echo "MutateNeuronTraitsProb 0"
echo "MutateLinkTraitsProb 0"
echo "MutateGenomeTraitsProb 0"

# Parameters received as input
for p in "${params[@]}"
do
    # Split the parameter and value by the '='
    psplit=(${p/=/ })
    param=${psplit[0]}
    value=${psplit[1]}
    case ${param} in
      Activation) # Convert activation function parameter
        echo | activation_functions $value;;
      InterspeciesCrossoverRate) # Convert from percentage to decimal
        value=$(echo "scale=5; ${value}/100" | bc)
        echo $param $value;;
      WeightMutationMaxProportion) # Dependent on MaxWeight
        mutweightprop=$value
        if [ ! -z ${maxweight+x} ]; then
          mutweightmax=$(echo "scale=2; ${maxweight}*${mutweightprop}" | bc)
          echo "WeightMutationMaxPower ${mutweightmax}"
          echo "WeightReplacementMaxPower ${mutweightmax}"
        fi;;
      MaxWeight) # Required for WeightMutationMaxProportion
        maxweight=$value
        if [ ! -z ${mutweightprop+x} ]; then
          mutweightmax=$(echo "scale=2; ${maxweight}*${mutweightprop}" | bc)
          echo "WeightMutationMaxPower ${mutweightmax}"
          echo "WeightReplacementMaxPower ${mutweightmax}"
        fi;;
      *) # Default case: print "param value"
        echo $param $value;;
    esac
done

# End tag
echo -e "\nNEAT_ParametersEnd"
