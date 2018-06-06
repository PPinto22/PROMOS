import csv
import random

from numpy import NaN


class Data:
    # If 'file_path' is specified, data is read from the file; else, it's copied from 'inputs' and 'targets'
    def __init__(self, file_path=None, input_labels=None, target_label='target', positive_class=1,
                 inputs=None, targets=None):
        self.inputs = list()  # List of input rows
        self.targets = list()  # List of target values
        self.positives = list()  # Indexes of positive cases in data
        self.negatives = list()  # Indexes of negative cases in data
        self.input_labels = input_labels  # List of input labels; values are stored in the same order as their labels
        self.target_label = target_label  # Label of the target column
        self.positive_class = positive_class  # Which target value is to be considered positive
        self.n_inputs = None  # Number of input columns

        if file_path is not None:
            self._init_from_file(file_path)
        else:
            self._init_from_data(inputs, targets)

    def _init_from_file(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            # Read fieldnames (aka labels)
            file_labels = list(next(reader))
            if self.input_labels is None:
                self.input_labels = [label for label in file_labels if label != self.target_label]
            else:
                if sorted(self.input_labels + [self.target_label]) != sorted(file_labels):
                    raise ValueError('The specified labels (input_labels + target_label) '
                                     'don\'t match those read from {}'.format(file_path))

            self.n_inputs = len(self.input_labels)

            # Map the order of labels in _file_labels to that specified in input_labels
            input_order = [NaN] * self.n_inputs
            for i, label in enumerate(self.input_labels):
                input_order[i] = file_labels.index(label)
            target_idx = file_labels.index(self.target_label)

            for i, row in enumerate(reader):
                if len(row) != self.n_inputs + 1:
                    raise ValueError('Row length mismatch')

                # Read row inputs
                inputs = [NaN] * self.n_inputs
                for j in range(len(inputs)):
                    inputs[j] = row[input_order[j]]

                # Read target
                target = row[target_idx]

                self._add_row(inputs, target, i)

    def _init_from_data(self, inputs, targets):
        if len(inputs) != len(targets):
            raise ValueError('The number of inputs does not match the number of targets')

        try:
            self.n_inputs = len(inputs[0])
        except IndexError:
            pass

        for i, (input, target) in enumerate(zip(inputs, targets)):
            if len(input) != self.n_inputs:
                raise ValueError('Row length mismatch')

            self._add_row(input, target, i)

    def _add_row(self, input, target, row_idx):
        self.inputs.append(input)
        positive_or_negative = self.positives if target == self.positive_class else self.negatives
        positive_or_negative.append(row_idx)
        self.targets.append(target)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]

    def is_balanced(self):
        return abs(len(self.positives) - len(self.negatives)) <= 1

    def get_sample(self, size, balanced=True, seed=None):
        random.seed(seed)

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

        positives_sample = random.sample(self.positives, positives_size)
        negatives_sample = random.sample(self.negatives, negatives_size)
        total_sample = sorted(positives_sample + negatives_sample)
        sample_inputs = [self.inputs[i] for i in total_sample]
        sample_targets = [self.targets[i] for i in total_sample]

        return Data(inputs=sample_inputs, targets=sample_targets, input_labels=self.input_labels,
                    target_label=self.target_label, positive_class=self.positive_class)
