import json
import subprocess
import pickle as cPickle
import os
import shutil
import sys
import time
import numpy as np


MAGENTA = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 


def load_pickle(fname):
    with open(fname) as f:
        return cPickle.load(f)


def write_pickle(o, fname):
    with open(fname, 'w') as f:
        cPickle.dump(o, f, -1)


def load_json(fname):
    with open(fname) as f:
        return json.loads(f.read())


def load_json_lines(fname):
    with open(fname) as f:
        for line in f:
            yield json.loads(line)


def write_json(d, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(d))


def lines(fname):
    with open(fname) as f:
        for line in f:
            yield line


def lines_in_file(fname):
    return int(subprocess.check_output(
        ['wc', '-l', fname]).strip().split()[0])


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rmkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def logged_loop(iterable, n=None):
    if n is None:
        n = len(iterable)
    step = max(1, n / 1000)
    prog = Progbar(n)
    for i, elem in enumerate(iterable):
        if i % step == 0 or i == n - 1:
            prog.update(i + 1)
        yield elem


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class NestedDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return self.setdefault(key, NestedDict())


# mostly stolen from keras
class Progbar(object):
    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, avg_values=[], exact_values=[]):
        for k, v in avg_values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact_values:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, avg_values=[], exact_values=[]):
        self.update(self.seen_so_far+n, avg_values, exact_values)
