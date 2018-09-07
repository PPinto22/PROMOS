import datetime
import os
import shutil
import traceback
from threading import Thread
from time import sleep

import util
import evolver
from data import Data


class Online(Thread):
    def __init__(self, evolver, shift, width, test_ratio=0.3):
        super(Online, self).__init__()
        self.evolver = evolver
        self.shift = shift
        self.width = width
        self.test_ratio = 0.3
        self.finished = False
        self.files_per_window = width//shift
        self.window_files = []

    def get_file_path(self, file_name, suffix=None):
        out_dir = self.evolver.options.out_dir if self.evolver.options.out_dir not in [None, 'NULL'] else '.'
        suffix = '.' + str(suffix) if suffix is not None else ''
        return '{}/{}{}'.format(out_dir, file_name, suffix)

    def extract_data(self):
        initial_time = util.datetime_to_string(datetime.datetime.now())
        temp_file_name = self.get_file_path(file_name=initial_time, suffix='tmp')
        self._extract_data(temp_file_name)
        end_time = util.datetime_to_string(datetime.datetime.now())
        data_file_name = self.get_file_path(file_name='collection__{}__{}'.format(initial_time, end_time), suffix='csv')
        self._prepare_data(temp_file_name, data_file_name)
        return data_file_name

    def _extract_data(self, outfile):
        # TODO!
        shutil.copy('../data/2weeks/best_idf_mini.csv', outfile)

    def _prepare_data(self, datafile, newname):
        # TODO!
        shutil.copy(datafile, newname)
        # Cleanup temp file
        if os.path.isfile(datafile):
            os.remove(datafile)

    def _split_data(self, data):
        if self.test_ratio == 0:
            return data, None
        else:
            return data.split((1-self.test_ratio, self.test_ratio))

    def add_data(self, file):
        self.window_files.append(file)
        if len(self.window_files) > self.files_per_window:
            del self.files_per_window[0]

    def load_dataset(self):
        data = Data(self.window_files)
        return self._split_data(data)

    def run(self):
        while not self.finished:
            try:
                new_data_file = self.extract_data()
                self.add_data(new_data_file)
                train, test = self.load_dataset()
                with self.evolver.gen_lock:
                    if not self.evolver.running:
                        with self.evolver.start_lock:
                            self.evolver.set_data(train, test, keep_old_if_none=True)
                            self.evolver.start_lock.notify()
                    else:
                        self.evolver.shift_window(new_train=train, new_test=test)
                # FIXME!
                sleep(30)
            except Exception as e:
                self.evolver.force_terminate = True
                self.finished = True
                self.evolver.log_error(traceback.format_exc())