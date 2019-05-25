import numpy as np
import pandas as pd
from util import add_noise, write_file, read_file, county_state_matrix, basic_data
from sklearn.ensemble import RandomForestRegressor

# Fit a random forest model, without using active learning.
# Returns: The fitted model.
# pct_polled: The proportion of data randomly chosen to be revealed.
#           Should be a decimal between 0 and 1. Defaults to 1.
def baseline(pct_polled=1):
    np.random.seed(1)

    x, y = basic_data()
    n_counties = x.shape[0]
    n_train = int(np.round(n_counties * pct_polled))
    train_subset = np.random.choice(n_counties, size=n_train, replace=False)
    x_train = x[train_subset, :]
    y_train = add_noise(y[train_subset, :])

    forest = RandomForestRegressor(n_estimators=100, max_depth=4,
                                   random_state=2, )
    forest.fit(x_train, y_train)
    return forest


def predict_all_counties(forest, save_path):
    predictions = forest.predict(x)
    sq_err = np.sqrt(np.square(predictions - y))
    full_x = read_file('../data/full_clean_data.csv')
    data = np.concatenate((full_x[0][:, :3], predictions, sq_err), axis=1)
    header = full_x[0][:7]
    header[3] = "pred_turnout"
    header[4] = "pred_dem"
    header[5] = "sq_err_turnout"
    header[6] = "sq_err_dem"
    write_file(save_path, header, data)