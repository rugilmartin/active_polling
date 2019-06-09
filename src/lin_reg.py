import numpy as np
import util
import results
from sklearn.linear_model import LinearRegression

def lin_reg():
    X, y = util.basic_data()
    polls = y
    polls = util.add_noise(y)
    party = LinearRegression()
    turnout = LinearRegression()
    data_percent = np.concatenate((range(4,30), range(30, 105, 5)))/100
    np.random.seed()

    square_errors = np.zeros([2,len(data_percent)])
    for i in range(len(data_percent)):
        for j in range(20):
            sample = np.random.choice(range(len(X)), size = int(data_percent[i]*len(X)), replace = False)
            X_train = X[sample]
            party.fit(X_train, polls[sample,0])
            turnout.fit(X_train,polls[sample,1])
            preds = np.column_stack((party.predict(X), turnout.predict(X)))
            square_error = (np.square(y - preds)).mean(axis = 0)
            square_errors[:,i] +=square_error
        square_errors[:,i] /= 20
    util.plot("lin_reg", data_percent, square_errors, legend = ["% democrat", "% turnout"], x_label = "% counties", y_label = "MSE", title = "MSE vs. %data")

lin_reg()

