import util
import lin_reg
import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def next_countys(solver, curr_labels, X, y):
    num_students = 8
    preds = np.zeros([num_students, len(X), 2])
    for s in range(num_students):
        student_data = np.random.choice(curr_labels, size = int(len(curr_labels)/2), replace = False)
        preds[s] = solver(X, X[student_data], y[student_data]) 
    var = np.var(preds, axis = 0).sum(axis = 1)
    uncertain = np.argsort(var)[::-1]
    points = []
    for point in uncertain:
        if point not in curr_labels:
            points.append(point)
        if len(points) == 5:
            return points

def committe(solver, solver_name, intervals, reps):
    np.random.seed()
    X, y = util.basic_data()
    polls = util.add_noise(y)
    curr_labels = np.random.choice(range(len(X)), size = 4, replace = False)
    X_train = X[curr_labels]
    square_errors = np.zeros([2, len(intervals)])
    for i in range(len(intervals)):
        print("interval: ", intervals[i])
        for j in range(reps):
            while len(curr_labels) <= intervals[i]:
                next_points = next_countys(solver, curr_labels, X, polls)
                curr_labels = np.append(curr_labels, next_points)
            curr_labels = curr_labels[:intervals[i]]
            preds = solver(X, X[curr_labels], polls[curr_labels])
            square_errors[:,i] += util.square_error(y, preds)
        square_errors[:,i] /= reps
    square_errors = np.vstack((square_errors.mean(axis = 0), util.performance(solver, intervals, reps).mean(axis = 0)))
    util.plot("committe", intervals/len(X), square_errors, legend = [solver_name, "random"], x_label = "% counties", y_label = "MSE", title = "Committe")


def main():
    intervals = np.array(range(10,100,10))
    reps = 100
    solver = lin_reg.regularized
    committe(solver, "lin_reg committe", intervals, reps)

main()

    

