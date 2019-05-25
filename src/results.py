import numpy as np
import csv
from util import read_file, write_file
import util


#preds should have shape (3140,2) with predicted dem_percent, and predicted turnout


def state_percent(preds):
    headers, data = read_file("../data/full_clean_data.csv")
    electoral_headers, electorate = read_file("../data/electoral.csv")
    percent = np.zeros(51)
    for i in range(len(percent)):
        state_name = electorate[i,0]
        counties = data[state_name == data[:,2]]
        state_preds = preds[state_name == data[:,2]]
        dems = (state_preds[:,0] * state_preds[:,1] * counties[:,3].astype(float)).sum()
        total = (state_preds[:,1] * counties[:,3].astype(float)).sum()
        percent[i] = dems/total
    return percent

def electoral_votes(preds):
    electoral_headers, electoral_votes = read_file("../data/electoral.csv")
    percent = state_percent(preds)
    print(percent)
    dem_electorates = np.around(np.around(percent) * electoral_votes[:,1].astype(int)).sum()
    return 538 - dem_electorates, dem_electorates
