#!/bin/bash

# Input example: PopulationSize=100 MutationRate=0.05
arguments=$*

# Read parameters into array
IFS=' ' read -r -a params <<< $arguments

echo "NEAT_ParametersStart"
# Print default parameters
echo "
MutateNeuronTraitsProb 0
MutateLinkTraitsProb 0
MutateGenomeTraitsProb 0
ActivationFunction_UnsignedGauss_Prob 0.333
ActivationFunction_UnsignedSigmoid_Prob 0.334
ActivationFunction_Relu_Prob 0.333
"

# Print parameters received as input
for p in "${params[@]}"
do
    # Replace '=' with a space and print
    echo "${p/=/ }"
done

echo "
NEAT_ParametersEnd"
