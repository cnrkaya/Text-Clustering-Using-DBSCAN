from custom_dbscan import DBSCAN

def dbscan_grid_search(X_data, lst, clst_count, eps_space = 0.5,
                       min_samples_space = 5, min_clust = 0, max_clust = 10):

    """
Performs a hyperparameter grid search for DBSCAN.

Parameters:
    * X_data            = data used to fit the DBSCAN instance
    * lst               = a list to store the results of the grid search
    * clst_count        = a list to store the number of non-whitespace clusters
    * eps_space         = the range values for the eps parameter
    * min_samples_space = the range values for the min_samples parameter
    * min_clust         = the minimum number of clusters required after each search iteration in order for a result to be appended to the lst
    * max_clust         = the maximum number of clusters required after each search iteration in order for a result to be appended to the lst

"""

    # Importing counter to count the amount of data in each cluster
    from collections import Counter


    # Starting a tally of total iterations
    n_iterations = 0


    # Looping over each combination of hyperparameters
    for eps_val in eps_space:
        for samples_val in min_samples_space:

            dbscan_grid = DBSCAN(eps=eps_val, min_samples=samples_val)


            # fit_transform
            clustering = dbscan_grid.fit(X = X_data)
            clusters =clustering.labels_


            # Counting the amount of data in each cluster
            cluster_count = Counter(clusters)


            # Saving the number of clusters
            n_clusters = sum(abs(pd.np.unique(clusters))) - 1

            print(n_clusters)


            # Increasing the iteration tally with each run of the loop
            n_iterations += 1


            # Appending the lst each time n_clusters criteria is reached
            if n_clusters >= min_clust and n_clusters <= max_clust:

                lst.append([eps_val,
                                        samples_val,
                                        n_clusters])


                clst_count.append(cluster_count)

    # Printing grid search summary information
    print(f"""Search Complete. \nYour list is now of length {len(lst)}. """)
    print(f"""Hyperparameter combinations checked: {n_iterations}. \n""")