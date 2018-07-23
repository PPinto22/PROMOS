import argparse
import csv
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import util
import encoder as enc

INPUTS_DTYPE = np.float32
TARGETS_DTYPE = np.uint8
TIMESTAMPS_DTYPE = 'datetime64[s]'


class Data:
    # If 'file_path' is specified, data is read from the file;
    # otherwise, it's copied from 'inputs', 'targets' and 'timestamps'
    def __init__(self, file_path=None, input_labels=None, target_label='target', positive_class=1,
                 timestamp_label='timestamp', inputs=None, targets=None, timestamps=None, is_sorted=False,
                 date_format='%Y-%m-%d %H:%M:%S', timestamps_only=False):
        self.inputs = list()  # list (converted to np.array later) of input rows
        self.targets = list()  # list (converted to np.array later) of target values
        self.timestamps = list()  # list (converted to np.array later) of timestamps
        self.positives = list()  # list of positive cases indexes
        self.negatives = list()  # list of negative cases indexes
        self.input_labels = input_labels  # List of input labels; values are stored in the same order as their labels
        self.target_label = target_label  # Label of the target column
        self.timestamp_label = timestamp_label  # Label of the timestamp column
        self.positive_class = positive_class  # Which target value is to be considered positive
        self.n_inputs = 0  # Number of input columns
        self.n_outputs = 1  # Number of outputs -- static, always 1
        self.has_timestamps = False  # Does the data contain timestamps
        self.is_sorted = is_sorted  # Is the data sorted by time
        self.date_format = date_format  # Date format to use with datetime.strptime
        self.timestamps_only = timestamps_only  # Contains timestamps only

        self.file_labels = None  # Header of the csv file
        self.input_order = None  # Indexes of the inputs columns in the csv file
        self.target_idx = None  # Index of the target column in the csv file
        self.timestamp_idx = None  # Index of the timestamp column in the csv file

        self.file_path = file_path
        if file_path is not None:
            self._init_from_file(file_path)
        else:
            self._init_from_data(inputs, targets, timestamps)

        self.order = None
        if self.has_timestamps and not self.is_sorted:
            self.sort()

    def encode_from_mapping(self, mapping):
        inputs_pd = pd.DataFrame(self.inputs, columns=self.input_labels)
        inputs_encoded = enc.Encoder.encode_from_mapping(inputs_pd, mapping)
        self.input_labels = list(inputs_encoded.columns)
        self.n_inputs = len(self.input_labels)
        self.inputs = inputs_encoded.values

    def encode(self, encoder):
        inputs_pd = pd.DataFrame(self.inputs, columns=self.input_labels)
        inputs_encoded, mapping = encoder.encode(inputs_pd, return_mapping=True)
        self.inputs = inputs_encoded.values
        return mapping

    @staticmethod
    def get_csv_reader(file):
        return csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

    def _init_from_file(self, file_path):
        with open(file_path, 'r') as file:
            reader = self.get_csv_reader(file)
            # Read fieldnames
            self.file_labels = list(next(reader))
            self.has_timestamps = self.timestamp_label in self.file_labels
            if self.input_labels is None:
                self.input_labels = [label for label in self.file_labels if
                                     label not in [self.target_label, self.timestamp_label]]
            file_input_target_labels = [x for x in self.file_labels if x != self.timestamp_label]
            if sorted(self.input_labels + [self.target_label]) != sorted(file_input_target_labels):
                raise ValueError('The specified input or target labels '
                                 'don\'t match those read from {}'.format(file_path))

            self.n_inputs = len(self.input_labels)

            # Map the order of labels in file_labels to that specified in input_labels
            self.input_order = np.zeros(self.n_inputs, dtype=int)
            for i, label in enumerate(self.input_labels):
                self.input_order[i] = self.file_labels.index(label)
            self.target_idx = self.file_labels.index(self.target_label)
            self.timestamp_idx = self.file_labels.index(self.timestamp_label) if self.has_timestamps else None

            for i, row in enumerate(reader):
                extra_cols = 2 if self.has_timestamps else 1
                if len(row) != self.n_inputs + extra_cols:
                    raise ValueError('Row length mismatch')

                if self.timestamps_only:
                    self.timestamps.append(datetime.strptime(row[self.timestamp_idx], self.date_format))
                else:
                    inputs, target, timestamp = self.parse_row(row)
                    self.add_row(inputs, target, timestamp, i)

        self._convert_to_numpy()

    def _init_from_data(self, inputs, targets, timestamps):
        length = len(inputs)
        if length != len(targets) or length != len(timestamps):
            raise ValueError('The number of rows of inputs, targets and timestamps do not match')

        try:
            self.n_inputs = len(inputs[0])
            self.has_timestamps = timestamps[0] is not None
        except IndexError:
            pass

        if not self.timestamps_only:
            self.inputs = inputs
            self.targets = targets
        self.timestamps = timestamps

    def parse_row(self, row):
        # inputs = np.zeros(self.n_inputs)
        inputs = [None] * self.n_inputs
        for j in range(len(inputs)):
            inputs[j] = row[self.input_order[j]]
        target = row[self.target_idx]
        timestamp = datetime.strptime(row[self.timestamp_idx], self.date_format) if self.has_timestamps else None
        return inputs, target, timestamp

    def add_row(self, input, target, timestamp, row_idx, use_numpy=False):
        if use_numpy:
            self.inputs[row_idx] = input
            self.targets[row_idx] = target
            self.timestamps[row_idx] = timestamp
        else:
            self.inputs.append(input)
            self.targets.append(target)
            self.timestamps.append(timestamp)

        positive_or_negative = self.positives if target == self.positive_class else self.negatives
        positive_or_negative.append(row_idx)

    def _convert_to_numpy(self):
        if self.input_labels is None:
            self.input_labels = list(range(self.n_inputs))

        if not self.timestamps_only:
            self.inputs = np.array(self.inputs)
            self.targets = np.array(self.targets, dtype=TARGETS_DTYPE)
        if self.timestamps:
            self.timestamps = np.array(self.timestamps)

    def sort(self):
        assert self.has_timestamps, "Cannot sort if there are no timestamps."
        order = np.argsort(self.timestamps)
        if not self.timestamps_only:
            self.inputs = self.inputs[order]
            self.targets = self.targets[order]
        self.timestamps = self.timestamps[order]
        self.order = order + 1  # Plus 1 for the header

        # Recreate the positive and negative lists
        for i, target in enumerate(self.targets):
            positive_or_negative = self.positives if target == self.positive_class else self.negatives
            positive_or_negative.append(i)

        self.is_sorted = True
        return self

    def __len__(self):
        if not self.timestamps_only:
            return len(self.inputs)
        else:
            return len(self.order)

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]

    def __bool__(self):
        return len(self) > 0

    def size(self):
        return len(self.inputs), self.n_inputs

    def is_balanced(self):
        return abs(len(self.positives) - len(self.negatives)) <= 1

    def get_time_range(self):
        if not self.has_timestamps:
            return None, None

        if self.is_sorted:
            return self.timestamps[0], self.timestamps[-1]
        else:
            return np.amin(self.timestamps), np.amax(self.timestamps)

    def get_num_inputs(self):
        if self.input_labels is not None:
            return len(self.input_labels)
        else:
            return len(self.inputs[0])

    # Returns the index of the first occurrence of a timestamp that is >= than dt
    def find_first_datetime(self, dt, start=0):
        assert self.is_sorted
        for i in range(start, len(self.timestamps)):
            if self.timestamps[i] >= dt:
                return i
        raise ValueError('No datetime >= {} was found'.format(str(dt)))

    # Returns the index of the last occurrence of a timestamp that is <= than dt
    def find_last_datetime(self, dt, start=0):
        assert len(self.timestamps) > 1
        assert self.is_sorted
        for i in range(start, len(self.timestamps) - 1):
            if self.timestamps[i + 1] > dt:
                return i + 1
        return len(self.timestamps) - 1

    def get_sample(self, size, balanced=True, seed=None):
        np.random.seed(seed)

        size = min(size, len(self))  # Assert that the sample size is at most equal to all the data

        if balanced:  # Try to get a 50/50 split of positives/negatives
            half_size = size // 2
            positives_size = min(half_size, len(self.positives))
            negatives_size = min(half_size, len(self.negatives))
            if positives_size < half_size:
                # Not enough positives for a balanced sample; compensate with extra negatives
                negatives_size += half_size - positives_size
            elif negatives_size < half_size:
                # Not enough negatives for a balanced sample; compensate with extra positives
                positives_size += half_size - negatives_size
        else:  # Mantain ratio of positives/negatives
            positives_size = (len(self.positives) * size) // len(self)
            negatives_size = (len(self.negatives) * size) // len(self)

        positives_sample = np.random.choice(self.positives, positives_size, replace=False) \
            if positives_size > 0 else np.array([], dtype=np.uint32)
        negatives_sample = np.random.choice(self.negatives, negatives_size, replace=False) \
            if negatives_size > 0 else np.array([], dtype=np.uint32)

        total_sample = np.concatenate((positives_sample, negatives_sample))
        total_sample.sort()  # To preserve order

        sample_inputs = self.inputs[total_sample]
        sample_targets = self.targets[total_sample]
        sample_timestamps = self.timestamps[total_sample]

        return Data(inputs=sample_inputs, targets=sample_targets, timestamps=sample_timestamps,
                    input_labels=self.input_labels, target_label=self.target_label,
                    timestamp_label=self.timestamp_label, positive_class=self.positive_class,
                    date_format=self.date_format, is_sorted=self.is_sorted)

    def split(self, probs, seed=None):
        np.random.seed(seed)
        choices = np.random.choice(range(len(probs)), len(self), replace=True, p=probs)
        splits = [[] for key in range(len(probs))]
        for idx, c in enumerate(choices):
            splits[c].append(idx)
        return (self.get_subset_by_indexes(indexes) for indexes in splits)

    def get_subset_by_indexes(self, indexes):
        inputs = self.inputs[indexes]
        targets = self.targets[indexes]
        timestamps = self.timestamps[indexes]

        return Data(inputs=inputs, targets=targets, timestamps=timestamps, input_labels=self.input_labels,
                    target_label=self.target_label, timestamp_label=self.timestamp_label,
                    positive_class=self.positive_class, date_format=self.date_format, is_sorted=self.is_sorted)

    def get_subset_by_time_interval(self, start, end):
        assert end > start
        assert start >= 0 and end < len(self)

        length = end - start + 1
        inputs = np.zeros((length, self.n_inputs))
        targets = np.zeros(length)
        timestamps = np.empty(length, dtype='datetime64[s]')

        if not self.timestamps_only:
            for i in range(length):
                inputs[i] = self.inputs[start + i]
                targets[i] = self.targets[start + i]
                timestamps[i] = self.timestamps[start + i]
        else:
            # Get file line numbers
            lines = set()
            for i in range(start, end + 1):
                lines.add(self.order[i])

            with open(self.file_path, 'r') as file:
                reader = self.get_csv_reader(file)
                count = 0
                for i, row in enumerate(reader):
                    if i in lines:
                        line_inputs, line_target, line_timestamp = self.parse_row(row)
                        inputs[count] = line_inputs
                        targets[count] = line_target
                        timestamps[count] = line_timestamp
                        count += 1
                    if count >= len(lines):
                        break

        return Data(inputs=inputs, targets=targets, timestamps=timestamps,
                    input_labels=self.input_labels, target_label=self.target_label,
                    timestamp_label=self.timestamp_label, positive_class=self.positive_class,
                    date_format=self.date_format, is_sorted=self.is_sorted)

    def save(self, file_path):
        util.make_dir(file_path=file_path)
        with open(file_path, 'w') as file:
            writer = csv.writer(file, csv.QUOTE_NONNUMERIC)
            header = ([self.timestamp_label] if self.has_timestamps else []) + [self.target_label] + self.input_labels
            writer.writerow(header)
            for inputs, target, timestamp in zip(self.inputs, self.targets, self.timestamps):
                row = ([timestamp] if self.has_timestamps else []) + [target] + list(inputs)
                writer.writerow(row)


class SlidingWindow(Data):
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
        self._window = 0  # Current window + 1

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
                # train_end_dt = t_start + self.train_width
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

    def get_window_data(self, window_i):
        if window_i < 0 or window_i > len(self.windows):
            raise IndexError('Invalid window index: {}'.format(window_i))
        window = self.windows[window_i]
        train = self.get_subset_by_time_interval(window[0], window[1])
        test = self.get_subset_by_time_interval(window[2], window[3]) if self.has_test else None

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
    parser.add_argument('data_file', help='path to data file to be encoded and/or sampled', metavar='DATA'),
    parser.add_argument('-o', '--outdir', dest='out_dir', default='.',
                        help='directory where to save output files', metavar='DIR')
    parser.add_argument('-v', '--val', dest='val', type=util.ratio, default=0, metavar='RATIO',
                        help='use this fraction of the data as a validation data-set')
    parser.add_argument('-t', '--test', dest='test', type=util.ratio, default=0, metavar='RATIO',
                        help='use this fraction of the data as a test data-set')
    parser.add_argument('-E', '--encoder', dest='encoder', metavar='FILE', default=None,
                        help='configuration file for numeric encoding. The encoding is performed over the training'
                             'data-set only. The test and validation data-sets are generated by mapping their raw'
                             'values to the corresponding codification in the training data-set')
    parser.add_argument('--id', dest='id', metavar='ID', default=None,
                        help='identifier used to name the output files (e.g., ID_train.csv, ID_val.csv, ID_test.csv)')
    parser.add_argument('--seed', dest='seed', metavar='S', type=util.uint, default=None,
                        help='specify an RNG integer seed')

    options = parser.parse_args()

    assert options.encoder is not None or options.val > 0 or options.test > 0

    util.make_dir(options.out_dir)
    if options.seed is not None:
        random.seed(options.seed)

    return options


if __name__ == '__main__':
    def file_name(suffix):
        if not has_split:
            suffix = 'encoded_data' if options.id is None else None

        return '{}/{}.csv'.format(options.out_dir, util.join_str('_', (options.id, suffix)))


    options = parse_args()

    encoder = enc.Encoder(options.encoder) if options.encoder is not None else None
    data = Data(options.data_file)
    train_size = 1 - options.val - options.test
    has_split = train_size < 1

    train, val, test = data.split((train_size, options.val, options.test), seed=options.seed)
    if encoder is not None:
        mapping = train.encode(encoder)
        val.encode_from_mapping(mapping)
        test.encode_from_mapping(mapping)

    train.save(file_name('train'))
    if val:
        val.save(file_name('val'))
    if test:
        test.save(file_name('test'))
