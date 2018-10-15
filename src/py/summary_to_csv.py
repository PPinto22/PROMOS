import csv
import json
import util
import re

from datetime import datetime
from collections import namedtuple

Run = namedtuple('Run', 'dir id label runs')

runs = [
    # Run('../../results/2wks_best_raw', 'best_raw', 'BEST NEAT RAW', 5),
    # Run('../../results/2wks_best_idf', 'best_idf', 'BEST NEAT IDF', 5),
    # Run('../../results/2wks_best_pcp', 'best_pcp', 'BEST NEAT PCP', 5),
    # Run('../../results/2wks_hn_best_raw', 'best_raw', 'BEST HNEAT RAW', 3),
    # Run('../../results/2wks_hn_best_idf', 'best_idf', 'BEST HNEAT IDF', 3),
    # Run('../../results/2wks_hn_best_pcp', 'best_pcp', 'BEST HNEAT PCP', 2),
    #
    # Run('../../results/2wks_test_raw', 'test_raw', 'TEST NEAT RAW', 3),
    # Run('../../results/2wks_test_idf', 'test_idf', 'TEST NEAT IDF', 10),
    # Run('../../results/2wks_test_pcp', 'test_pcp', 'TEST NEAT PCP', 10),
    # Run('../../results/2wks_hn_test_raw', 'test_raw', 'TEST HNEAT RAW', 3),
    # Run('../../results/2wks_hn_test_idf', 'test_idf', 'TEST HNEAT IDF', 3),
    # Run('../../results/2wks_hn_test_pcp', 'test_pcp', 'TEST HNEAT PCP', 2)

    Run('../../results/2wks_best_raw_quick', 'best_raw', 'BEST NEAT RAW', 2),
    Run('../../results/2wks_best_idf_quick', 'best_idf', 'BEST NEAT IDF', 2),
    Run('../../results/2wks_best_pcp_quick', 'best_pcp', 'BEST NEAT PCP', 2),
    Run('../../results/2wks_test_raw_quick', 'test_raw', 'TEST NEAT RAW', 2),
    Run('../../results/2wks_test_idf_quick', 'test_idf', 'TEST NEAT IDF', 2),
    Run('../../results/2wks_test_pcp_quick', 'test_pcp', 'TEST NEAT PCP', 2),

    Run('../../results/2wks_hn_best_raw_quick', 'best_raw', 'BEST HNEAT RAW', 2),
    Run('../../results/2wks_hn_best_idf_quick', 'best_idf', 'BEST HNEAT IDF', 2),
]

OUT_FILE = 'summary.csv'


def time_to_minutes(x):
    if isinstance(x, str):
        x_split = re.split(' days?, ', x)
        days = int(x_split[0]) if len(x_split) > 1 else 0
        x = x_split[1] if len(x_split) > 1 else x_split[0]

        return days*24*60 + \
               ((datetime.strptime(x, "%H:%M:%S.%f") - datetime.strptime("00:00", "%H:%M")).total_seconds() / 60.0)
    else:
        return x


def read_summary(run, i):
    multi_run_str = '({})'.format(i) if run.runs is not None else ''
    with open('{}/{}{}_summary.json'.format(run.dir, run.id, multi_run_str, i + 1), 'r') as file:
        data = json.load(file)
        run_time = time_to_minutes(data['run_time'])
        eval_time = time_to_minutes(data['eval_time'])
        ea_time = time_to_minutes(data['ea_time'])
        ea_eval_time = eval_time + ea_time
        ea_perc = ea_time / ea_eval_time * 100
        train_auc = data['best']['fitness']
        test_auc = data['best']['fitness_test']
        return train_auc, test_auc, run_time, eval_time, ea_time, ea_eval_time, ea_perc


def read_summaries(run):
    run_range = list(range(run.runs)) if run.runs is not None else [1]
    return [read_summary(run, i+1) for i in run_range]


if __name__ == '__main__':
    ds = []
    for run in runs:
        n_runs = run.runs if run.runs is not None else 1
        summaries = read_summaries(run)
        summaries_avg = util.map_avg(summaries)
        summaries_avg = [run.label, n_runs] + summaries_avg
        ds.append(summaries_avg)

    with open(OUT_FILE, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['type', 'runs', 'train_auc', 'test_auc',
                         'run_time', 'eval_time', 'ea_time', 'ea_eval_time', 'ea_time_%'])
        for row in ds:
            writer.writerow(row)
