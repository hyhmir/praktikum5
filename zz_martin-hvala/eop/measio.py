import os
import glob
import subprocess
import numpy as np

def read_measdict(globpattern, sep=','):
    '''Import all measurements from a directory of measurement files. Each
    file is read into a numpy array (column-first) and added to the measurement
    dict (keys are filenames with extentions removed).'''
    def ignored(row):
        return (str.isspace(row) or str.lstrip(row)[0] == '#')

    measdict = {}
    for fname in glob.glob(globpattern):
        with open(fname) as file:
            meas = [
                [float(el) for el in row.rstrip().split(sep)]
                for row in file.readlines() if not ignored(row)
            ]
            name, _ = str.rsplit(os.path.basename(fname), '.', maxsplit=1)
            measdict[name] = np.array(meas).T
    return measdict
