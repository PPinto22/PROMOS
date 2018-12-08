import numpy as np

if __name__ == '__main__':
    INPUTS = 11
    HIDDEN = 4
    OUT = '/home/pedro/Desktop/nnet_genome.txt'

    with open(OUT, 'w') as file:
        file.write('GenomeStart 0\n')
        # Inputs
        for i in range(INPUTS):
            file.write('Neuron {} 1 0 1 0 0 0 0\n'.format(i))
        # Output
        file.write('Neuron {} 4 1 1 1 0 0 0\n'.format(INPUTS))
        # Hidden
        for i in range(HIDDEN):
            file.write('Neuron {} 3 0.5 1 1 0 0 0\n'.format(i + INPUTS + 1))

        # Links Input -> Hidden
        for i in range(INPUTS):
            for j in range(HIDDEN):
                link_index = i*HIDDEN + j
                file.write('Link {} {} {} 0 {}\n'.format(i, INPUTS+1+j, link_index, np.random.random()))
        # Links Hidden -> Output
        for i in range(HIDDEN):
            file.write('Link {} {} {} 0 {}\n'.format(i + INPUTS + 1, INPUTS, INPUTS*HIDDEN+i, np.random.random()))
        file.write('GenomeEnd\n')

