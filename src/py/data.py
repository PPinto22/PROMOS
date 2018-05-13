import csv
import random


class Data:
    def __init__(self, file_path=None, data=None, target_label='target', positive_class=1):
        self.data = list()
        self.targets = list()  # The 'target' column in data
        self.positives = list()  # Indexes of positive cases in data
        self.negatives = list()  # Indexes of negative cases in data
        self.target_label = target_label
        self.positive_class = positive_class

        if file_path is not None:
            file = open(file_path)
            data = csv.DictReader(file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)

        for i, row in enumerate(data):
            self.data.append(row)
            target = row[target_label]
            positive_or_negative = self.positives if target == positive_class else self.negatives
            positive_or_negative.append(i)
            self.targets.append(target)

        if file_path is not None:
            file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def is_balanced(self):
        return abs(len(self.positives) - len(self.negatives)) <= 1

    def get_sample(self, size, balanced=True, seed=None):
        random.seed(seed)

        size = min(size, len(self.data))  # Assert that the sample size is at most equal to all the data

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
            positives_size = (len(self.positives) * size) // len(self.data)
            negatives_size = (len(self.negatives) * size) // len(self.data)

        positives_sample = random.sample(self.positives, positives_size)
        negatives_sample = random.sample(self.negatives, negatives_size)
        total_sample = sorted(positives_sample + negatives_sample)
        sample_data = [self.data[i] for i in total_sample]

        return Data(data=sample_data, target_label=self.target_label, positive_class=self.positive_class)
