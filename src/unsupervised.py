import util
import numpy as np
from sklearn.cluster import KMeans
import lin_reg

def points_from_clusters(X, num_points):
    k_means = KMeans(n_clusters = num_points).fit(X)
    labels = k_means.labels_
    clusters = np.array([np.argwhere(labels == c) for c in range(num_points)])
    return np.array([np.random.choice(clusters[i].flatten()) for i in range(num_points)])

def main():
    X, y = util.basic_data()
    reps = 10
    intervals = np.array(range(20,300,20))
    square_errors = util.performance(lin_reg.regularized, intervals, reps, chooser = points_from_clusters)
    square_errors = np.vstack((square_errors.mean(axis = 0), util.performance(lin_reg.regularized, intervals, reps).mean(axis = 0)))
    util.plot("kmeans", intervals/len(X), square_errors, legend = ["kmeans", "random"], title = "Kmeans")

main()

