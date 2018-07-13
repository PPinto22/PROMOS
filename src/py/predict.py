import argparse
import csv
from MultiNEAT import Genome

import substrate
import util
from evaluator import Evaluator
from data import Data, SlidingWindow


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
    parser.add_argument('-W', '--window', dest='width', metavar='W', type=util.uint, default=None,
                        help='Sliding window width (train + test) in hours')
    parser.add_argument('-w', '--test-window', dest='test_width', metavar='W', type=util.uint, default=None,
                        help='Test sliding window width in hours')
    parser.add_argument('-S', '--shift', dest='shift', metavar='S', type=util.uint, default=None,
                        help='Sliding window shift in hours')
    parser.add_argument('--no-inputs', dest='inputs', action='store_false', help='Do not save input variables to file')

    args = parser.parse_args()
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

if __name__ == '__main__':
    args = parse_args()
    if not args.out_file.endswith('.csv'):
        args.out_file = args.out_file + '.csv'

    genome = Genome(args.genome_file)
    subst = substrate.load_substrate(args.substrate_file) if args.substrate_file is not None else None
    network = util.build_network(genome, args.method, substrate)
    data = Data(args.data_file)

    slider = SlidingWindow(args.width, args.shift, args.test_width,
                           file_path=args.data_file) if args.width is not None else None

    util.make_dir(file_path=args.out_file)
    if slider is None:
        predictions = Evaluator.predict(network, data.inputs)
        write_predictions(data.inputs, data.targets, predictions, args.out_file, include_inputs=args.inputs)
    else:
        for i, (train_data, test_data) in enumerate(slider):
            print("[Window {}/{}] {} rows...".format(i, slider.n_windows, len(test_data)))
            predictions = Evaluator.predict(network, test_data.inputs)
            write_predictions(test_data.inputs, test_data.targets, predictions, args.out_file, i,
                              include_inputs=args.inputs)
