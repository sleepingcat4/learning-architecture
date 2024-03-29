import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3) -> int:
        self.k = k
        self.centroids = None
    
    @staticmethod
    def euclidean_distance(data_point, centroids):
        np.sqrt(np.sum(centroids - data_point)**2, axis=1)

    def fit(self, X, max_iterations=200):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1])),
# X is the data
# y are the clusters
        for _ in range(max_iterations):
            y = []
            for data_point in X:
                distances = KMeans.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

            ## finding the optimum cluster points
            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
            
            # reposition clusters
            cluster_centres = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centres.append(self.centroids[i])
                else:
                    cluster_centres.append(np.mean(X[indices], axis=0)[0])
                
            if np.max(self.centroids - np.array(cluster_centres)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centres)
            
        return y

