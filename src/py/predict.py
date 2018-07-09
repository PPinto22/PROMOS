import argparse
import csv
from MultiNEAT import Genome

import substrate
import util
from evaluator import Evaluator
from data import Data


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('genome_file', help='path to genome file', metavar='GENOME')
    methods = ['neat', 'hyperneat']
    parser.add_argument('data_file', help='path to data file', metavar='DATA'),
    parser.add_argument('out_file', help='save predictions to FILE', metavar='FILE')
    parser.add_argument('-m', '--method', dest='method', metavar='M', choices=methods, default='neat',
                        help='which algorithm was used to generate the network: ' + ', '.join(methods))
    parser.add_argument('-s', '--substrate', dest='substrate_file', metavar='S', default=None,
                        help='path to a substrate; required if method is hyperneat')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.out_file.endswith('.csv'):
        args.out_file = args.out_file + '.csv'

    genome = Genome(args.genome_file)
    subst = substrate.load_substrate(args.substrate_file) if args.substrate_file is not None else None
    network = util.build_network(genome, args.method, substrate)
    data = Data(args.data_file)

    predictions = Evaluator.predict(network, data.inputs)

    with open(args.out_file, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(data.input_labels + [data.target_label] + ['prediction'])
        for inputs, target, pred in zip(data.inputs, data.targets, predictions):
            row = list(inputs) + [target] + [pred]
            writer.writerow(row)


