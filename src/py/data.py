import csv
import random
from datetime import datetime
import numpy as np


class Data:
    # If 'file_path' is specified, data is read from the file;
    # otherwise, it's copied from 'inputs', 'targets' and 'timestamps'
    def __init__(self, file_path=None, input_labels=None, target_label='target', positive_class=1,
                 timestamp_label='timestamp', inputs=None, targets=None, timestamps=None, is_sorted=False,
                 date_format='%Y-%m-%d %H:%M:%S'):
        self.inputs = None  # np.array of input rows
        self.targets = None  # np.array of target values
        self.timestamps = None  # np.array of timestamps
        self.positives = None  # list of positive cases indexes
        self.negatives = None  # list of negative cases indexes
        self.input_labels = input_labels  # List of input labels; values are stored in the same order as their labels
        self.target_label = target_label  # Label of the target column
        self.timestamp_label = timestamp_label  # Label of the timestamp column
        self.positive_class = positive_class  # Which target value is to be considered positive
        self.n_inputs = 0  # Number of input columns
        self.has_timestamps = False  # Does the data contain timestamps
        self.is_sorted = is_sorted  # Is the data sorted by time
        self.date_format = date_format  # Date format to use with datetime.strptime

        if file_path is not None:
            self._init_from_file(file_path)
        else:
            self._init_from_data(inputs, targets, timestamps)

        if self.has_timestamps and not is_sorted:
            self.sort()

    def _init_from_file(self, file_path):
        self._init_arrays(list)
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            # Read fieldnames
            file_labels = list(next(reader))
            self.has_timestamps = self.timestamp_label in file_labels
            if self.input_labels is None:
                self.input_labels = [label for label in file_labels if
                                     label not in [self.target_label, self.timestamp_label]]
            file_input_target_labels = [x for x in file_labels if x != self.timestamp_label]
            if sorted(self.input_labels + [self.target_label]) != sorted(file_input_target_labels):
                raise ValueError('The specified input or target labels '
                                 'don\'t match those read from {}'.format(file_path))

            self.n_inputs = len(self.input_labels)

            # Map the order of labels in _file_labels to that specified in input_labels
            input_order = np.zeros(self.n_inputs, dtype=int)
            for i, label in enumerate(self.input_labels):
                input_order[i] = file_labels.index(label)
            target_idx = file_labels.index(self.target_label)
            timestamp_idx = file_labels.index(self.timestamp_label) if self.has_timestamps else None

            for i, row in enumerate(reader):
                extra_cols = 2 if self.has_timestamps else 1
                if len(row) != self.n_inputs + extra_cols:
                    raise ValueError('Row length mismatch')

                # Read row inputs
                inputs = np.zeros(self.n_inputs)
                for j in range(len(inputs)):
                    inputs[j] = row[input_order[j]]
                # Read target
                target = row[target_idx]
                # Read timestamp if exists
                timestamp = datetime.strptime(row[timestamp_idx], self.date_format) if self.has_timestamps else None

                self._add_row(inputs, target, timestamp, i)

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

        self._init_arrays(np.ndarray, rows=length, input_columns=self.n_inputs)
        for i, (input, target, timestamp) in enumerate(zip(inputs, targets, timestamps)):
            if len(input) != self.n_inputs:
                raise ValueError('Row length mismatch')

            self._add_row(input, target, timestamp, i, use_numpy=True)

    def _add_row(self, input, target, timestamp, row_idx, use_numpy=False):
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

    def _init_arrays(self, list_type, rows=None, input_columns=None):
        if list_type is list:
            self.inputs = list()
            self.targets = list()
            self.timestamps = list()
        elif list_type is np.ndarray:
            if rows is None or rows < 0:
                raise AttributeError('Invalid number of rows: ' + str(rows))
            self.inputs = np.zeros((rows, input_columns))
            self.targets = np.zeros(rows)
            self.timestamps = np.empty(rows, dtype='datetime64[s]')
        else:
            raise AttributeError('Invalid type: ' + str(list_type))
        self.positives = list()
        self.negatives = list()

    def _convert_to_numpy(self):
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
        self.timestamps = np.array(self.timestamps)

    def sort(self):
        assert self.has_timestamps, "Cannot sort if there are no timestamps."

        order = np.argsort(self.timestamps)
        self.inputs = self.inputs[order]
        self.targets = self.targets[order]
        self.timestamps = self.timestamps[order]

        # Recreate the positive and negative lists
        for i, target in enumerate(self.targets):
            positive_or_negative = self.positives if target == self.positive_class else self.negatives
            positive_or_negative.append(i)

        self.is_sorted = True
        return self

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]

    def is_balanced(self):
        return abs(len(self.positives) - len(self.negatives)) <= 1

    def get_num_inputs(self):
        if self.input_labels is not None:
            return len(self.input_labels)
        else:
            return len(self.inputs[0])

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

        positives_sample = np.random.choice(self.positives, positives_size, replace=False)
        negatives_sample = np.random.choice(self.negatives, negatives_size, replace=False)

        total_sample = np.concatenate((positives_sample, negatives_sample))
        total_sample.sort()  # To preserve order

        sample_inputs = self.inputs[total_sample]
        sample_targets = self.targets[total_sample]
        sample_timestamps = self.timestamps[total_sample]

        return Data(inputs=sample_inputs, targets=sample_targets, timestamps=sample_timestamps,
                    input_labels=self.input_labels, target_label=self.target_label,
                    timestamp_label=self.timestamp_label, positive_class=self.positive_class,
                    date_format=self.date_format, is_sorted=self.is_sorted)
