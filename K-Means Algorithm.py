import matplotlib.pyplot as plt
import numpy as np
import copy
'''This is a K-means algorithm that takes in a data set and a number of clusters and returns the clusters and centroids'''
'''The first section accepts user inputs for the number of data points, number of clusters, minimum and maximum values, and threshold value'''
num = int(input('Enter many data points would you like (recommended under 10,000): '))
K = int(input('Enter many clusters would you like (recommended under 40 unless very large data set): '))
minimum = float(input('Enter the minimum value of the data points: '))
maximum = float(input('Enter the maximum value of the data points: '))
threshold = float(input(f'Please enter the threshold value to stop the algorithm (suggested around .5-2% of maximum value): '))

'''This section creates the data set and plots it'''
z = []
for i in range(num):
    z.append([np.random.uniform(minimum, maximum), np.random.uniform(minimum, maximum)])
Y = copy.deepcopy(z)
for point in range(len(Y)):
    Y[point].append([])
X = np.array(z)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], s = 40, color= 'k')
plt.show()

'''This function returns the distance between two points but also works for arrays'''
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

'''This function takes the data set and centroids and plots both on the same graph'''
def plot_progress(X, C, centroids, i, counter):
    flag = True
    # print(f'Counter is {counter}')
    if i == 0 and counter == 1:
        plt.scatter(X[:, 0], X[:, 1], c = 'k', s = 40)
        # print(f'np.arange(len(centroids)) is {np.arange(len(centroids))}')
        plt.scatter(centroids[:, 0], centroids[:, 1], c = np.arange(len(centroids)), s = 100, cmap = plt.cm.brg, marker='x')
        plt.show() 
    if i == 0 and flag == True: 
        plt.scatter(X[:, 0], X[:, 1], c = C, s = 40, cmap = plt.cm.brg)
        plt.scatter(centroids[:, 0], centroids[:, 1], c = np.arange(len(centroids)), s = 100, cmap = plt.cm.brg, marker='x')
        plt.show()


'''This function finds the min distance of a point to a centroid'''
def closest_centroid(x, centroids):
    distances = [distance(x, centroid) for centroid in centroids]
    closest = distances.index(min(distances))
    return closest

'''This function assigns a point to a cluster that it is closest to'''
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        clusters.append(closest_centroid(point, centroids))
    return clusters, data

'''This function updates the centroids by taking the mean of the cluster'''
def update_centroids(data, clusters, K):
    new_centroids = []
    for i in range(K):
        new_centroids.append(list(np.mean(data[np.array(clusters) == i], axis=0)))
    # print(f'The new centroids are given as: {new_centroids}')
    return np.array(new_centroids)
    
'''This function is the parent function and calls the other functions and initializes the data set/centroids'''
def k_means(X, K, i, counter):
    clusters = []
    for k in range(K):
        clusters.append([np.random.uniform(minimum, maximum), np.random.uniform(minimum,maximum)])
    centroids = np.array(clusters)
    # print(f'The initial centroids are: {centroids}')
    closest = []
    # C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
    # print(f'C is given as {C}')
    C = []
    for x_i in X:
        dist_list = []
        for y_k in centroids:
            dist_list.append(distance(x_i, y_k))
        # print(dist_list)
        C.append(np.argmin(dist_list))
    C = np.array(C)
    # print(f'C is given as {C}')

    plt.xlabel(f'The graph for iteration {1}')
    assign_clusters(X, centroids)      
    plot_progress(X, C, centroids, i, counter)  
    return np.array(centroids), C

'''This function runs k_means and creates the iterations then shows the final plot'''
def kmeans(i):
    counter = 1
    test = k_means(X, K, i, counter)
    # print(f'Test')
    centroids = test[0]
    C = list(test[1])
    # print(Y)
    # print(C)
    for point in range(len(Y)):
        Y[point][-1] = C[point]
    # print(f'\nThe new data set is given as {Y}')
    prev_centroid = []
    for k in range(K):
        prev_centroid.append([-999999999, -99999999])

    cur_centroid = centroids
    is_significant = compare_centroids(prev_centroid, cur_centroid, threshold)

    '''This while loop compares the average distance to the threshold or if the iterations are greater than 10
    If it is then it creates another graph and updates centroids'''
    # print(f'is this true? {is_significant > threshold}')
    while is_significant > threshold and counter < 10:
        # print(f'This is iteration {counter}')
        counter += 1
        plt.xlabel(f'The graph for iteration {counter}')
        clusters, data = assign_clusters(X, centroids)
        prev_centroid = copy.deepcopy(centroids)
        centroids = update_centroids(X, clusters, K)
        cur_centroid = copy.deepcopy(centroids)
        # print(f'The input centroids for iteration {counter} is {centroids}')
        C = []
        for x_i in X:
            dist_list = []
            for y_k in centroids:
                dist_list.append(distance(x_i, y_k))
            C.append(np.argmin(dist_list))
        C = np.array(C)
        # C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        plot_progress(X, C, centroids, i, counter)
        is_significant = compare_centroids(prev_centroid, cur_centroid, threshold)
        # print(f'The is significant value for iteration {counter} is given as: {is_significant}')
    x_val = minimum*1.2
    y_val = maximum*.9
    togetherness = centroid_distance(X, centroids)
    plt.scatter(X[:, 0], X[:, 1], c = C, s = 40, cmap = plt.cm.brg)
    # print(f'The centroids for run {i} are {centroids}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c = np.arange(len(centroids)), s = 100, cmap = plt.cm.brg, marker='x')
    plt.xlabel(f'The final graph which is given as iteration {counter}')
    plt.title(f'The graph has a togetherness of {togetherness}', fontsize = 9)
    return centroids, Y, togetherness

'''This function compares the previous and current centroids and returns the average distance'''
def compare_centroids(prev_centroid, cur_centroid, threshold):
    cur_dist = []
    for i in range(K):
        cur_dist.append(np.sqrt(np.sum((prev_centroid[i] - cur_centroid[i])**2)))
    is_significant = sum(cur_dist)/K
    return is_significant

'''Write a function that finds the average distance of all points to the centroid it is closest to'''
def centroid_distance(data, centroids):
    distances = []
    for point in data:
        distances.append(distance(point, centroids[closest_centroid(point, centroids)]))
    togetherness = sum(distances)/len(distances)
    return togetherness

'''This function runs the functions 9 times and creates a new figure for the final run'''
def running():
    data_list = []
    togetherness_list = []
    for i in range(9):
        plt.figure(i+1)
        run_centroids, data, togetherness = kmeans(i)
        data_list.append(data)
        togetherness_list.append(togetherness)
    plt.show()
    min_index = togetherness_list.index(min(togetherness_list))
    final_data = data_list[min_index]
    best_graph = min_index+1
    print(f'The minimum togetherness is {min(togetherness_list)} for graph {best_graph} and the data set is {data_list[min_index]}')
    final_data_set = open(r'data.txt', 'w')
    final_data_set.write(f'{final_data}')
    final_data_set.close()
run_it = running()
