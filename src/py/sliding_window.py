import argparse
import random
from datetime import timedelta

import data
import util
import encoder as enc


class SlidingWindow(data.Data):
    # Widths and shift in hours
    def __init__(self, width, shift, test_width, **kwargs):
        self.test_width = test_width if test_width is not None else 0
        assert test_width >= 0
        assert all(x > 0 for x in (width, shift))
        assert test_width <= width

        super().__init__(timestamps_only=True, **kwargs)

        assert self.has_timestamps

        self.has_test = test_width > 0
        self.width = timedelta(hours=width)
        self.shift = timedelta(hours=shift)
        self.test_width = timedelta(hours=test_width)
        self.train_width = self.width - self.test_width

        # List of index tuples: (train_begin, train_end, test_begin, test_end)
        # or (train_begin, train_end), if there is no test data
        self.windows = list()
        self.setup_windows()
        self.n_windows = len(self.windows)
        self._window = 0  # Current window

    def setup_windows(self):
        global_start, global_end = self.get_time_range()  # Start and end datetimes

        t_start = global_start  # Trial window start time
        t_end = min(t_start + self.width, global_end)  # Trial window end time
        t_width = t_end - t_start  # Trial window width
        # While the trial window's width is at least 80% of the regular width, accept it
        while t_width > 0.8 * self.width:
            # Find indexes based on the datetime limits
            train_start = self.find_first_datetime(t_start)
            if self.has_test:
                train_end_dt = t_end - self.test_width
                train_end = self.find_last_datetime(train_end_dt, start=train_start)
                test_start = train_end + 1
                test_end = self.find_last_datetime(t_end, start=test_start)
                window = (train_start, train_end, test_start, test_end)
            else:
                train_end = self.find_last_datetime(t_end, start=train_start)
                window = (train_start, train_end)
            self.windows.append(window)

            # Next window
            t_start = t_start + self.shift
            t_end = min(t_start + self.width, global_end)
            t_width = t_end - t_start

    def get_window_data(self, window_i, update_state=False):
        if window_i < 0 or window_i > len(self.windows):
            raise IndexError('Invalid window index: {}'.format(window_i))
        window = self.windows[window_i]
        train = self.get_subset_by_index_range(window[0], window[1])
        test = self.get_subset_by_index_range(window[2], window[3]) if self.has_test else None

        if update_state:
            self._window = window_i + 1
        return train, test

    def get_current_window_data(self):
        return self.get_window_data(self._window)

    def get_current_window_index(self):
        return self._window - 1

    def __getitem__(self, item):
        return self.get_window_data(item)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            train, test = self.get_current_window_data()
        except IndexError:
            raise StopIteration
        self._window += 1
        return train, test

    def has_next(self):
        return self._window < self.n_windows

    def reset(self):
        self._window = 0


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_file', help='path to data file to be encoded and/or divided by windows', metavar='DATA'),
    parser.add_argument('-o', '--outdir', dest='out_dir', default='.',
                        help='directory where to save output files', metavar='DIR')
    parser.add_argument('-W', '--window', dest='width', metavar='W', type=util.ufloat, default=120,
                        help='Sliding window width (train + test) in hours')
    parser.add_argument('-w', '--test-window', dest='test_width', metavar='W', type=util.ufloat, default=24,
                        help='Test sliding window width in hours')
    parser.add_argument('-S', '--shift', dest='shift', metavar='S', type=util.ufloat, default=24,
                        help='Sliding window shift in hours')
    parser.add_argument('-E', '--encoder', dest='encoder', metavar='FILE', default=None,
                        help='configuration file for numeric encoding. The encoding is performed over the training'
                             'data-set only. The test and validation data-sets are generated by mapping their raw'
                             'values to the corresponding codification in the training data-set')
    parser.add_argument('--id', dest='id', metavar='ID', default='windows',
                        help='identifier used to name the output files '
                             '(e.g., ID_train(1).csv, ID_test(1).csv, where 1 is the window index)')
    parser.add_argument('--seed', dest='seed', metavar='S', type=util.uint, default=None,
                        help='specify an RNG integer seed')

    options = parser.parse_args()

    util.make_dir(options.out_dir)
    if options.seed is not None:
        random.seed(options.seed)

    return options


if __name__ == '__main__':
    def file_name(window, suffix=None):
        return '{}/{}({}).csv'.format(options.out_dir, util.join_str('_', (options.id, suffix)), window)

    options = parse_args()
    has_test = options.test_width > 0
    encoder = enc.Encoder(options.encoder) if options.encoder is not None else None
    slider = SlidingWindow(options.width, options.shift, options.test_width, file_path=options.data_file)

    col_order = None
    for i, (train, test) in enumerate(slider):
        if encoder is not None:
            mapping = train.encode(encoder, soft_order=col_order)
            if test is not None:
                test.encode_from_mapping(mapping)
            col_order = train.input_labels

        train.save(file_name(window=i, suffix='train' if has_test else None))
        has_test and test.save(file_name(window=i, suffix='test'))
