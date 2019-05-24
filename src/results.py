import numpy as np
import csv
from data_clean import read_file, write_file


#preds should have shape (3140,2) with predicted dem_percent, and predicted turnout


def electoral_votes(preds):
    electoral_headers, electoral_votes = read_file("../data/electoral.csv")
    headers, data = read_file("../data/full_clean_data.csv")
    state_percent = np.zeros(51)
    for i in range(len(state_percent)):
        state_name = electoral_votes[i,0]
        counties = data[state_name == data[:,2]]
        state_preds = preds[state_name == data[:,2]]
        dems = (state_preds[:,0] * state_preds[:,1] * counties[:,3].astype(float)).sum()
        total = (state_preds[:,1] * counties[:,3].astype(float)).sum()
        state_percent[i] = dems/total
    dem_electorates = np.around(np.around(state_percent) * electoral_votes[:,1].astype(int)).sum()
    return 538 - dem_electorates, dem_electorates

#incomplete, might want later
'''
def votes_off(preds):
    electoral_headers, electoral_votes = read_file("../data/electoral.csv")
    headers, data = read_file("../data/full_clean_data.csv")
    state_percent = np.zeros(51)
    diff = [0, 0]
    for i in range(len(state_percent)):
        state_name = electoral_votes[i,0]
        counties = data[state_name == data[:,2]]
        state_preds = preds[state_name == data[:,2]]
        dems_pred = (state_preds[:,0] * state_preds[:,1] * counties[:,3].astype(float))
        reps_pred = ((1-state_preds[:,0]) * state_preds[:,1] * counties[:,3].astype(float))
        reps_actual = counties[:,13].astype(int)
        dems_actual = counties[:,14].astype(int)
        print(reps_pred, dems_pred, reps_actual, dems_actual)
        diff[0] += reps_pred - reps_actual
        diff[1] += dems_pred - dems_actual
    return np.around(np.array(diff))
'''
