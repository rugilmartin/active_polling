import util
import lin_reg
import random_forest
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def next_counties(curr_labels, X, y):
    forest = RandomForestRegressor(n_estimators=50, max_depth=4)
    forest.fit(X[curr_labels], y[curr_labels])
    tree_preds = np.zeros((forest.n_estimators, len(X), 2))
    for s in range(forest.n_estimators):
        tree_preds[s] = forest.estimators_[1].predict(X)
    var = np.var(tree_preds, axis=0).sum(axis=1)
    uncertain = np.argsort(var)[::-1]
    points = []
    for point in uncertain:
        if point not in curr_labels:
            points.append(point)
        if len(points) == 5:
            return points


def entropy(intervals, reps):
    np.random.seed()
    X, y = util.basic_data()
    polls = util.add_noise(y)
    curr_labels = np.random.choice(range(len(X)), size=intervals[0], replace=False)
    # X_train = X[curr_labels]
    square_errors = np.zeros([2, len(intervals)])
    for j in range(reps):
        preds = random_forest.train_predict(X, X[curr_labels], polls[curr_labels])
        square_errors[:, 0] += util.square_error(y, preds)
    square_errors[:, 0] /= reps
    for i in range(1, len(intervals)):
        print("interval: ", intervals[i])
        for j in range(reps):
            while len(curr_labels) <= intervals[i]:
                next_points = next_counties(curr_labels, X, polls)
                curr_labels = np.append(curr_labels, next_points)
            curr_labels = curr_labels[:intervals[i]]
            preds = random_forest.train_predict(X, X[curr_labels], polls[curr_labels])
            square_errors[:, i] += util.square_error(y, preds)
        square_errors[:, i] /= reps
    mse = square_errors.mean(axis=0)
    return (intervals/len(X), mse)


def plot_entropy(solver_name, percent_intervals, square_errors):
    util.plot(solver_name + "_entropy", percent_intervals, square_errors, legend=[solver_name + " entropy", "random"], x_label="% counties",
              y_label="MSE", title="Committee")


def main():
    intervals = np.array(range(10, 100, 10))

    reps = 5  # Choose 100 for linear regression, 5 for random forest
    (plt_X, mse_rf) = entropy(intervals, reps)
    perf_rf = util.performance(random_forest.train_predict, intervals, reps).mean(axis=0)
    plot_entropy("random_forest", plt_X, np.vstack((mse_rf, perf_rf)))



if __name__ == "__main__": main()


