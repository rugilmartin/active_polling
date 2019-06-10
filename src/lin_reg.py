import numpy as np
import util
import results
from sklearn.linear_model import Ridge, LinearRegression


def basic(X, X_train, y_train):
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict(X)
    return preds

def regularized(X, X_train, y_train, alpha = 0.1):
    clf = Ridge(alpha = alpha)
    clf.fit(X_train, y_train)
    preds = clf.predict(X)
    return preds

def performance(intervals, reps, solver, alpha = 0.1):
    X, y = util.basic_data()
    polls = util.add_noise(y)
    square_errors = np.zeros([2,len(intervals)])
    for i in range(len(intervals)):
        for j in range(reps):
            sample = np.random.choice(range(len(X)), size = intervals[i], replace = False)
            X_train = X[sample]
            y_train = polls[sample]
            if solver == "basic":
                preds = basic(X, X_train, y_train)
            if solver == "regularized":
                preds = regularized(X, X_train, y_train, alpha)
            square_errors[:,i] += util.square_error(y, preds)
        square_errors[:,i] /= reps
    return square_errors


def main():
    reps = 30
    intervals = np.array(range(100,500,20))
    square_errors = np.vstack((performance(intervals, reps, "basic").mean(axis = 0),
        performance(intervals, reps, "regularized", alpha = 1).mean(axis = 0),
        performance(intervals, reps, "regularized", alpha = 0.1).mean(axis = 0),
        performance(intervals, reps, "regularized", alpha = 0.01).mean(axis = 0)))
    util.plot("lin_reg", intervals, square_errors, legend = ["basic", "1", "0.1", "0.01"], x_label = "% counties", y_label = "MSE", title = "MSE vs. %data")

