import numpy as np
import csv
from util import read_file, write_file
import util


actual_state_votes = [0,0,0,0,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1,1,0,0,0]

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

def analyze_results(preds):
    percent = state_percent(preds)
    headers, data = read_file("../data/full_clean_data.csv")
    states = util.states()
    wrongly_dem = [states[i] for i in range(51) if (preds[i] > 0.5 and actual_state_votes[i] == 0)]
    wrongly_rep = [states[i]  for i in range(51) if (preds[i] <= 0.5 and actual_state_votes[i] == 1)]
    print("Wrongly predicted rep: ", wrongly_rep)
    print("Wrongly predicted dem: ", wrongly_dem)
    print("Electoral votes: ", electoral_votes(preds))
