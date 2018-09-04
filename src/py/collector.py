import datetime
import os
import shutil
from threading import Thread
from time import sleep

import util
from data import Data


class DataCollector(Thread):

    def __init__(self, evolver, duration=1, test_ratio=0.3):
        assert 0 <= test_ratio <= 0.8
        assert duration > 0

        super(DataCollector, self).__init__()
        self.evolver = evolver
        self.duration = duration
        self.test_ratio = test_ratio
        self.finished = False
        self.data_files = []

    def get_file_path(self, file_name, suffix=None):
        out_dir = self.evolver.options.out_dir if self.evolver.options.out_dir not in [None, 'NULL'] else '.'
        suffix = '.' + str(suffix) if suffix is not None else ''
        return '{}/{}{}'.format(out_dir, file_name, suffix)

    def extract_data(self, outfile):
        # TODO!
        shutil.copy('../data/2weeks/best_idf_mini.csv', outfile)

    def prepare_data(self, datafile, newname):
        # TODO!
        shutil.copy(datafile, newname)
        # Cleanup temp file
        if os.path.isfile(datafile):
            os.remove(datafile)

    def split_data(self, data):
        if self.test_ratio == 0:
            return data, None
        else:
            return data.split((1-self.test_ratio, self.test_ratio))

    def get_data(self):
        initial_time = util.datetime_to_string(datetime.datetime.now())
        temp_file_name = self.get_file_path(file_name=initial_time, suffix='tmp')
        self.extract_data(temp_file_name)
        end_time = util.datetime_to_string(datetime.datetime.now())
        data_file_name = self.get_file_path(file_name='collection__{}__{}'.format(initial_time, end_time), suffix='csv')
        self.prepare_data(temp_file_name, data_file_name)
        self.data_files.append(data_file_name)

        data = Data(data_file_name)
        return self.split_data(data)

    def run(self):
        while not self.finished:
            train, test = self.get_data()
            with self.evolver.start_lock:
                if self.evolver.generation == 0:
                    self.evolver.set_data(train, test, keep_old_if_none=True)
                else:
                    self.evolver.shift_window(new_train=train, new_test=test)
                self.evolver.start_lock.notify()
            # FIXME!
            sleep(30)