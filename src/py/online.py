import datetime
import os
import subprocess
import shutil
import traceback
from threading import Thread

import util
from data import Data

class Online(Thread):
    R_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../r')
    EXTRACT_SCRIPT = 'extract.R'
    TRANSFORM_SCRIPT = 'transform.R'
    PREP_SCRIPT = 'prep.R'

    def __init__(self, evolver, shift, width, test_ratio=0.3):
        super(Online, self).__init__()
        self.evolver = evolver
        self.shift = shift
        self.width = width
        self.test_ratio = test_ratio
        self.finished = False
        self.files_per_window = width//shift
        self.window_files = []

    def get_file_path(self, file_name):
        out_dir = self.evolver.options.out_dir if self.evolver.options.out_dir not in [None, 'NULL'] else '.'
        return os.path.abspath('{}/{}'.format(out_dir, file_name))

    def extract_data(self):
        initial_time = util.datetime_to_string(datetime.datetime.now())
        # Call extract.R
        subprocess.check_output(['Rscript', Online.EXTRACT_SCRIPT,
                                 '-w', self.get_file_path('etl'),
                                 '-s', 'sales.json',
                                 '-r', 'redis.json',
                                 '-t', str(self.shift)], cwd=Online.R_DIR)
        end_time = util.datetime_to_string(datetime.datetime.now())
        # Call transform.R
        subprocess.check_output(['Rscript', Online.TRANSFORM_SCRIPT,
                                 '-w', self.get_file_path('etl'),
                                 '-s', 'sales.json',
                                 '-r', 'redis.json',
                                 '-o', 'treated.json'], cwd=Online.R_DIR)
        # Call prep.R
        file_name = self.get_file_path('data/collection__{}__{}.csv'.format(initial_time, end_time))
        subprocess.check_output(['Rscript', Online.PREP_SCRIPT,
                                 '-f', self.get_file_path('etl/treated.json'),
                                 '-o', file_name], cwd=Online.R_DIR)
        return file_name

    def split_data(self, data):
        if self.test_ratio == 0:
            return data, None
        else:
            return data.split((1-self.test_ratio, self.test_ratio))

    def add_data(self, file):
        self.window_files.append(file)
        if len(self.window_files) > self.files_per_window:
            del self.window_files[0]

    def load_dataset(self):
        data = Data(self.window_files)
        return self.split_data(data)

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
            except Exception as e:
                self.evolver.force_terminate = True
                self.finished = True
                print(e)
                self.evolver.log_error(traceback.format_exc())
                with self.evolver.start_lock:
                    self.evolver.start_lock.notify()