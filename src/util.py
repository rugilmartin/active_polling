import numpy as np
import csv
import matplotlib.pyplot as plt

def read_file(in_filename):
    with open(in_filename) as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = [row for row in reader]
    return headers, np.array(data)

def write_file(out_filename, headers, data):
    with open(out_filename, 'w') as out_file:
        writer = csv.writer(out_file, delimiter = ',')
        writer.writerow(headers)
        writer.writerows(data)

def county_state_matrix():
    headers, data = read_file('../data/full_clean_data.csv')
    result = np.empty([len(data), 50])
    states = np.unique(data[:,2])[:-1]
    for i in range(len(states)):
        result[:,i] = (data[:,2] == states[i])
    return result.astype(float)

#returns x, y as floats
def basic_data():
    headers, data = read_file('../data/full_clean_data.csv')
    y = data[:,[15,16]].astype(float)
    x = np.delete(data, [0, 1, 2, 13,14,15,16], axis = 1).astype(float)
    x = np.concatenate((x, county_state_matrix()), axis = 1)
    return x, y

def states():
    headers, data = read_file('../data/full_clean_data.csv')
    return np.unique(data[:,2])

#party_var and bias comes from https://5harad.com/papers/polling-errors.pdf
#turnout_var and bias is just an estimate
def add_noise(y, party_var = 0.02, turnout_var = 0.04, party_bias = 0, turnout_bias = 0):
    y[:,0] += np.random.normal(party_bias, party_var, len(y))
    y[:,1] += np.random.normal(party_bias, turnout_bias, len(y))
    return y

def square_error(preds, y):
    return (np.square(y - preds)).mean(axis = 0)

def performance(solver, intervals, reps):
    X, y = basic_data()
    polls = add_noise(y)
    square_errors = np.zeros([2, len(intervals)])
    for i in range(len(intervals)):
        for j in range(reps):
            sample = np.random.choice(range(len(X)), size = intervals[i], replace = False)
            X_train = X[sample]
            y_train = polls[sample]
            preds = solver(X, X_train, y_train)
            square_errors[:, i] += square_error(y, preds)
        square_errors[:,i] /= reps
    return square_errors

def plot(savename, x, y, legend = None, x_label = None, y_label = None, title = None):
    for i in range(len(y)):
        plt.plot(x, y[i])   
    if legend != None:
        plt.legend(legend)
    if x_label != None:
        plt.xlabel(x_label)
    if y_label != None:
        plt.ylabel(y_label)
    if title != None:
        plt.title(title)
    plt.savefig("./output/" + savename) 

