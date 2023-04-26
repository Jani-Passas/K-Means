import matplotlib.pyplot as plt
import numpy as np
import copy
num = int(input('Enter many data points would you like: '))
K = int(input('Enter many clusters would you like:  '))
minimum = float(input('Enter the minimum value of the data points: '))
maximum = float(input('Enter the maximum value of the data points: '))
threshold = float(input(f'Please enter the threshold value to stop the algorithm (suggested around 1% of maximum value): '))
z = []
counter = 1
for i in range(num):
    z.append([np.random.uniform(minimum, maximum), np.random.uniform(minimum, maximum)])
Y = copy.deepcopy(z)
X = np.array(z)
# print(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 40, color= 'k')
plt.show()

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

'''This function takes the data set and centroids and plots both on the same graph'''
def plot_progress(X, C, centroids): 
    plt.scatter(X[:, 0], X[:, 1], c = C, s = 40, cmap = plt.cm.Spectral)
    plt.scatter(centroids[:, 0], centroids[:, 1], c = np.arange(len(centroids)), s = 100, cmap = plt.cm.Spectral, marker='x')
    plt.show()

def closest_centroid(x, centroids):
    distances = [distance(x, centroid) for centroid in centroids]
    closest = distances.index(min(distances))
    return closest
    
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        clusters.append(closest_centroid(point, centroids))
    return clusters, data

def update_centroids(data, clusters, K):
    new_centroids = []
    for i in range(K):
        new_centroids.append(list(np.mean(data[np.array(clusters) == i], axis=0)))
    # print(f'The new centroids are given as: {new_centroids}')
    return np.array(new_centroids)
    
def k_means(X, K):
    clusters = []
    for k in range(K):
        clusters.append([np.random.randint(minimum, maximum), np.random.randint(minimum,maximum)])
    clusters = np.array(clusters)
    # print(f'The clusters are given as: {clusters}')
    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    # print(f'The initial centroids are: {centroids}')
    closest = []
    C = []
    for x_i in X:
        dist_list = []
        for y_k in centroids:
            C_i = distance(x_i, y_k)
        C.append(np.array([np.argmin(C_i)]))
    # print(f'C is given as {C}')
    plt.scatter(X[:, 0], X[:, 1], c = C, s = 40, cmap = plt.cm.Spectral)
    plt.scatter(centroids[:, 0], centroids[:, 1], c = np.arange(len(centroids)), s = 100, cmap = plt.cm.Spectral, marker='x')
    plt.show()
    C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
    plt.xlabel(f'The graph for iteration {1}')
    assign_clusters(X, centroids)      
    plot_progress(X, C, centroids)  
    return np.array(centroids), C
    

test = k_means(X, K)
# print(f'Test')
centroids = test[0]
C = list(test[1])
# print(Y)
# print(C)
for point in range(len(Y)):
    Y[point].append(C[point])
print(f'\nThe new data set is given as {Y}')
prev_centroid = []
for k in range(K):
    prev_centroid.append([-999999999, -99999999])

cur_centroid = centroids
def compare_centroids(prev_centroid, cur_centroid, threshold):
    cur_dist = []
    for i in range(K):
        cur_dist.append(np.sqrt(np.sum((prev_centroid[i] - cur_centroid[i])**2)))
    is_significant = sum(cur_dist)/K
    return is_significant

is_significant = compare_centroids(prev_centroid, cur_centroid, threshold)
print(f'The is initial significant value is given as: {is_significant}')

while is_significant > threshold and counter < 10:
    counter += 1
    plt.xlabel(f'The graph for iteration {counter}')
    clusters, data = assign_clusters(X, centroids)
    prev_centroid = copy.deepcopy(centroids)
    centroids = update_centroids(X, clusters, K)
    cur_centroid = copy.deepcopy(centroids)
    # print(f'The input centroids for iteration {counter} is {centroids}')
    C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
    plot_progress(X, C, centroids)
    is_significant = compare_centroids(prev_centroid, cur_centroid, threshold)
    print(f'The is significant value for iteration {counter} is given as: {is_significant}')