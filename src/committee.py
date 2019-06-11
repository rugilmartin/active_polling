import util
import lin_reg
import random_forest
import numpy as np


def next_counties(solver, curr_labels, X, y):
    num_students = 8
    preds = np.zeros([num_students, len(X), 2])
    for s in range(num_students):
        student_data = np.random.choice(curr_labels, size=int(len(curr_labels) / 2), replace=False)
        preds[s] = solver(X, X[student_data], y[student_data])
    var = np.var(preds, axis=0).sum(axis=1)
    uncertain = np.argsort(var)[::-1]
    points = []
    for point in uncertain:
        if point not in curr_labels:
            points.append(point)
        if len(points) == 5:
            return points


def committee(solver, intervals, reps):
    np.random.seed()
    X, y = util.basic_data()
    polls = util.add_noise(y)
    curr_labels = np.random.choice(range(len(X)), size=4, replace=False)
    X_train = X[curr_labels]
    square_errors = np.zeros([2, len(intervals)])
    for j in range(reps):
        preds = random_forest.train_predict(X, X[curr_labels], polls[curr_labels])
        square_errors[:, 0] += util.square_error(y, preds)
    square_errors[:, 0] /= reps
    for i in range(1, len(intervals)):
        print("interval: ", intervals[i])
        for j in range(reps):
            while len(curr_labels) <= intervals[i]:
                next_points = next_counties(solver, curr_labels, X, polls)
                curr_labels = np.append(curr_labels, next_points)
            curr_labels = curr_labels[:intervals[i]]
            preds = solver(X, X[curr_labels], polls[curr_labels])
            square_errors[:, i] += util.square_error(y, preds)
        square_errors[:, i] /= reps
    mse = square_errors.mean(axis=0)
    return (intervals/len(X), mse)


def plot_committee(solver_name, percent_intervals, square_errors):
    util.plot(solver_name + "_committee", percent_intervals, square_errors, legend=[solver_name + " committee", "random"], x_label="% counties",
              y_label="MSE", title="Committee")


def main():
    intervals = np.array(range(10, 100, 10))

    reps = 5  # Choose 100 for linear regression, 5 for random forest
    rf = random_forest.train_predict
    (plt_X, mse_rf) = committee(rf, intervals, reps)
    perf_rf = util.performance(rf, intervals, reps).mean(axis=0)
    plot_committee("random_forest", plt_X, np.vstack((mse_rf, perf_rf)))

    reps = 100
    lr = lin_reg.regularized
    (plt_X, mse_lr) = committee(lr, intervals, reps)
    perf_lr = util.performance(lr, intervals, reps).mean(axis=0)
    plot_committee("lin_reg", plt_X, np.vstack((mse_lr, perf_lr)))

    util.plot("comparison_committee", plt_X, np.vstack((mse_rf, mse_lr)), legend=["random forest", "linear regression"], x_label="% counties",
              y_label="MSE", title="Committee")



if __name__ == "__main__": main()


