import util
import lin_reg
import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def next_countys(curr_labels, X, y):
    num_students = 8
    preds = np.zeros([num_students, len(X), 2])
    for s in range(num_students):
        student_data = np.random.choice(curr_labels, size = int(len(curr_labels)/2), replace = False)
        preds[s] = lin_reg.regularized(X, X[student_data], y[student_data]) 
    var = np.var(preds, axis = 0).sum(axis = 1)
    uncertain = np.argsort(var)[::-1]
    points = []
    for point in uncertain:
        if point not in curr_labels:
            points.append(point)
        if len(points) == 5:
            return points

def main():
    np.random.seed()
    reps = 100
    X, y = util.basic_data()
    polls = util.add_noise(y)
    intervals = np.array(range(15, 100, 10))
    curr_labels = np.random.choice(range(len(X)), size = 4, replace = False)
    X_train = X[curr_labels]
    square_errors = np.zeros([2, len(intervals)])
    for i in range(len(intervals)):
        print("interval: ", intervals[i])
        for j in range(reps):
            while len(curr_labels) <= intervals[i]:
                next_points = next_countys(curr_labels, X, y)
                curr_labels = np.append(curr_labels, next_points)
            curr_labels = curr_labels[:intervals[i]]
            preds = lin_reg.regularized(X, X[curr_labels], y[curr_labels])
            square_errors[:,i] += util.square_error(y, preds)
        square_errors[:,i] /= reps
    square_errors = np.vstack((square_errors.mean(axis = 0), lin_reg.performance(intervals, reps, "regularized").mean(axis = 0)))
    util.plot("committe", intervals/len(X), square_errors, legend = ["committe", "random"], x_label = "% counties", y_label = "MSE", title = "Committe")

main()

    

