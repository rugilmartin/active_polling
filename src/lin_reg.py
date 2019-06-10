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

def main():
    reps = 30
    intervals = np.array(range(100,500,20))
    square_errors = np.vstack((performance(intervals, reps, "basic").mean(axis = 0),
        performance(intervals, reps, "regularized", alpha = 1).mean(axis = 0),
        performance(intervals, reps, "regularized", alpha = 0.1).mean(axis = 0),
        performance(intervals, reps, "regularized", alpha = 0.01).mean(axis = 0)))
    util.plot("lin_reg", intervals, square_errors, legend = ["basic", "1", "0.1", "0.01"], x_label = "% counties", y_label = "MSE", title = "MSE vs. %data")

