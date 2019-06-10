import numpy as np
import pandas as pd
from util import plot, add_noise, write_file, read_file, county_state_matrix, basic_data
from sklearn.ensemble import RandomForestRegressor
import results

# Fit a random forest model, without using active learning.
# Returns: The fitted model.
# pct_polled: The proportion of data randomly chosen to be revealed.
#           Should be a decimal between 0 and 1. Defaults to 1.
def baseline(pct_polled=1):
    x, y = basic_data()
    n_counties = x.shape[0]
    n_train = int(np.round(n_counties * pct_polled))
    train_subset = np.random.choice(n_counties, size=n_train, replace=False)
    x_train = x[train_subset, :]
    y_train = add_noise(y[train_subset, :])

    forest = RandomForestRegressor(n_estimators=50, max_depth=4,
                                   random_state=2)
    forest.fit(x_train, y_train)
    return forest

def forest_train_predict(X, X_train, y_train):
    forest = RandomForestRegressor(n_estimators=50, max_depth=4,
                                   random_state=2)
    forest.fit(X_train, y_train)
    preds = forest.predict(X)
    return preds

def predict_and_record(forest, save_path):
    x, y = basic_data()
    predictions = forest.predict(x)
    sq_err = np.square(predictions - y)
    full_x = read_file('../data/full_clean_data.csv')
    data = np.concatenate((full_x[1][:, :3], predictions, sq_err), axis=1)
    header = full_x[0][:7]
    header[3] = "pred_dem"
    header[4] = "pred_turnout"
    header[5] = "sq_err_dem"
    header[6] = "sq_err_turnout"
    write_file(save_path, header, data)


def main():
    np.random.seed(1)
    mse = np.zeros((10, 2))
    x, y = basic_data()
    percentages = [1, 2, 3, 4, 5, 10, 20, 40, 60, 100]

    for i in percentages:
        pct_polled = i/100
        forest = baseline(pct_polled)
        save_path = "output/random_forest/baseline_" + str(i) + ".csv"
        predict_and_record(forest, save_path)
        mse[i-1:] = np.mean(np.square(forest.predict(x) - y), axis=0)
        print(i, "% trial complete")

    percentages = np.array(percentages).reshape((10, 1))
    data = np.concatenate((percentages, mse), axis=1)
    write_file("output/random_forest/baseline_mse.csv",
               ["percent_counties_polled", "mse_dem", "mse_turnout"], data)

    plot("/random_forest/mse_plot_50_4.png",
         percentages.reshape(10,),
         np.transpose(mse),
         legend=["MSE of % Democrat", "MSE of % Turnout"],
         x_label="Percent of Counties Polled",
         y_label="Mean Squared Error",
         title="MSE for Random Forest Baseline")


if __name__ == "__main__": main()