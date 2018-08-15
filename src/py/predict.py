import argparse
import csv
from MultiNEAT import Genome

from tabulate import tabulate

import substrate
import util
from data import Data
from encoder import Mapping
from evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('genome_file', help='path to genome file', metavar='GENOME')
    methods = ['neat', 'hyperneat']
    parser.add_argument('-o', '--out-file', dest='out_file', default='predictions.csv', metavar='FILE',
                        help='save predictions to FILE')
    parser.add_argument('-t', '--test', dest='test_file', default=None,
                        help='path to test data file', metavar='FILE')
    parser.add_argument('-m', '--method', dest='method', metavar='M', choices=methods, default='neat',
                        help='which algorithm was used to generate the network: ' + ', '.join(methods))
    parser.add_argument('-s', '--substrate', dest='substrate_file', metavar='S', default=None,
                        help='path to a substrate; required if method is hyperneat')
    parser.add_argument('-M', '--mapping', dest='mapping_file', metavar='FILE', default=None,
                        help='load the encoding mapping in the binary FILE')
    parser.add_argument('-W', '--window', dest='width', metavar='W', type=util.uint, default=None,
                        help='Sliding window width (train + test) in hours')
    parser.add_argument('-w', '--test-window', dest='test_width', metavar='W', type=util.uint, default=None,
                        help='Test sliding window width in hours')
    parser.add_argument('-S', '--shift', dest='shift', metavar='S', type=util.uint, default=None,
                        help='Sliding window shift in hours')
    parser.add_argument('--no-inputs', dest='inputs', action='store_false', help='Do not save input variables to file')
    parser.add_argument('--interactive', dest='interactive', action='store_true', help='interactive mode')
    parser.add_argument('--show-encoding', dest='show_encoding', action='store_true',
                        help='in interactive mode, print the input encodings')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                        help='Do not print any messages to stdout, except for the result')
    args = parser.parse_args()

    assert util.xor(args.test_file is not None,
                    args.interactive), 'Either specify a test data-set or choose interactive mode'

    return args


def write_predictions(inputs, targets, predictions, file_name, window=None, include_inputs=True):
    if window is not None and window == 0:
        with open(file_name, 'w') as file:
            file.truncate()

    with open(file_name, 'a') as file:
        writer = csv.writer(file, delimiter=',')
        header = (['window'] if window is not None else []) + \
                 (data.input_labels if include_inputs else []) + [data.target_label] + ['prediction']
        writer.writerow(header)
        for inputs, target, pred in zip(inputs, targets, predictions):
            row = ([window] if window is not None else []) + \
                  (list(inputs) if include_inputs else []) + [target] + [pred]
            writer.writerow(row)


def interactive():
    if not args.quiet:
        print('Format: values separated by \',\' with no spaces. Input Q to quit.')
        mapping and not None and print('Input order: {}'.format(','.join(mapping.col_names_raw)))
    while True:
        row = input('> ')
        if row.lower() == 'q':
            break
        row = row.split(',')
        try:
            if mapping is not None:
                row = mapping.map(row)
                if args.show_encoding:
                    print(tabulate([row], headers=mapping.col_names_encoded))
            pred = Evaluator.predict_single(network, row)
            print(pred)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    args = parse_args()

    if args.out_file is not None:
        if not args.out_file.endswith('.csv'):
            args.out_file = args.out_file + '.csv'

    genome = Genome(args.genome_file)
    subst = substrate.load_substrate(args.substrate_file) if args.substrate_file is not None else None
    network = util.build_network(genome, args.method, substrate)
    data = Data(args.test_file) if args.test_file is not None else None
    mapping = Mapping.load(args.mapping_file) if args.mapping_file is not None else None
    mapping is not None and data is not None and data.encode_from_mapping(mapping)
    util.make_dir(file_path=args.out_file)

    if args.interactive:
        interactive()
    else:
        predictions = Evaluator.predict(network, data.inputs)
        write_predictions(data.inputs, data.targets, predictions, args.out_file, include_inputs=args.inputs)
