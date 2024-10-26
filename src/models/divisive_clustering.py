import logging
from joblib import Parallel, delayed, Logger
import numpy as np

from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

from src.dependence_measures.bivariate import maximal_information_coefficient as mic


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]-[%(name)s.%(funcName)s:%(lineno)d] - %(levelname)s - %(message)s") 


class DivisiveHierarchicalClustering:
    """
    A class to perform divisive hierarchical clustering on data with multiple input and one output variable.
    """

    def __init__(self,
                 X,
                 Y,
                 min_cluster_size=10,
                 split_method='spectral',
                 split_use_XY=True,
                 max_depth=None):
        """
        Initializes the clustering algorithm with data and parameters.

        Parameters:
        - X: ndarray of shape (n_samples, n_features_X)
            Input variables values.
        - Y: ndarray of shape (n_samples, 1)
            Output variable values.
        - min_cluster_size: int, optional
            Minimum number of samples in a cluster to consider splitting further.
        - split_method: str, optional
            Clustering method for splitting.
        - split_use_XY: bool, optional
            Whether to use X and Y for splitting.
        - max_depth: int or None, optional
            Maximum depth of the clustering hierarchy.
        """
        self.X = X
        self.Y = Y
        self.min_cluster_size = min_cluster_size
        self.split_method = split_method
        self.split_use_XY = split_use_XY
        self.max_depth = max_depth
        self.clusters = []

    def fit(self):
        """
        Performs the divisive hierarchical clustering.
        """
        n_samples = self.X.shape[0]
        initial_indices = np.arange(n_samples)
        initial_cluster = {
            'indices': initial_indices,
            'depth': 0,
            'mic_metrics': self.compute_mic_metrics(self.X[initial_indices], self.Y[initial_indices])
        }
        queue = [initial_cluster]

        while queue:
            current_cluster = queue.pop(0)
            indices = current_cluster['indices']
            depth = current_cluster['depth']

            logging.info(f"Processing cluster at depth {depth:2d} with {len(indices):5d} samples")

            # Check stopping criteria
            if len(indices) <= self.min_cluster_size or (self.max_depth is not None and depth >= self.max_depth):
                self.clusters.append(current_cluster)
                continue

            # Attempt to split the cluster
            split_candidates = self.split_cluster(self.X[indices], self.Y[indices])

            if split_candidates is not None:
                best_candidate = None
                max_score = -np.inf

                # Evaluate each candidate split in parallel
                mic_metrics_current = current_cluster['mic_metrics']

                def evaluate_candidate(candidate):
                    labels = candidate['labels']
                    # Split indices based on labels
                    indices_left = indices[labels == 0]
                    indices_right = indices[labels == 1]

                    score, mic_metrics_left, mic_metrics_right = self.compute_split_score(
                        X_cluster_left=self.X[indices_left],
                        Y_cluster_left=self.Y[indices_left],
                        X_cluster_right=self.X[indices_right],
                        Y_cluster_right=self.Y[indices_right],
                        mic_metrics_current=mic_metrics_current
                    )

                    logging.info(f"left: {mic_metrics_left}, right: {mic_metrics_right}, current: {mic_metrics_current}, score: {score}")

                    return {
                        'score': score,
                        'indices_left': indices_left,
                        'indices_right': indices_right,
                        "mic_metrics_left": mic_metrics_left,
                        "mic_metrics_right": mic_metrics_right
                    }

                # # Parallel computation of candidate evaluations
                # results = Parallel(n_jobs=4)(
                #     delayed(evaluate_candidate)(candidate) for candidate in split_candidates
                # )
                results = [evaluate_candidate(candidate) for candidate in split_candidates]

                # Collect scores and find the candidate with the maximum score
                scores = np.array([result['score'] for result in results])
                max_index = np.argmax(scores)
                max_score = scores[max_index]
                best_result = results[max_index]
                best_candidate = best_result

                logging.info(f"Candidates obtained scores {scores}, with max_score {max_score:.3f} at index {max_index}")

                # Use the accept_split method to decide whether to accept the best split
                if best_candidate and self.accept_split(best_candidate["score"],
                                                        best_candidate['indices_left'],
                                                        best_candidate['indices_right']):
                    logging.info(f"Best candidate accepted")

                    # Add child clusters to the queue
                    queue.append({
                        'indices': best_candidate['indices_left'],
                        'depth': depth + 1,
                        'mic_metrics': best_candidate['mic_metrics_left'],
                    })
                    queue.append({
                        'indices': best_candidate['indices_right'],
                        'depth': depth + 1,
                        'mic_metrics': best_candidate['mic_metrics_right'],
                    })
                    continue  # Proceed to next cluster
                else:
                    logging.info(f"Best candidate rejected")

            # If split not accepted, add current cluster to final clusters
            self.clusters.append(current_cluster)

    def split_cluster(self, X_cluster, Y_cluster):
        """
        Attempts to split a given cluster using the specified clustering method.

        Parameters:
        - X_cluster: ndarray of shape (n_samples_cluster, n_features_X)
            Input variable values in the cluster.
        - Y_cluster: ndarray of shape (n_samples_cluster, n_features_Y)
            Output variable values in the cluster.

        Returns:
        - result: dict or None
            If successful, returns dict with key 'labels' (ndarray of cluster labels).
            Returns None if the cluster cannot be split.
        """
        n_samples = X_cluster.shape[0]
        if n_samples < 2:
            return None  # Not enough points to split

        if self.split_use_XY:
            # Combine X_cluster and Y_cluster without reshaping
            X_split = np.hstack([X_cluster, Y_cluster])
        else:
            X_split = X_cluster

        candidates = []

        if self.split_method == 'spectral':

            X_split_scaled = StandardScaler().fit_transform(X_split)

            for gamma in [0.5, 1.0, 10.0]:

                logging.info(f"Running spectral clustering with gamma={gamma}")

                clustering = SpectralClustering(n_clusters=2,
                                                assign_labels="kmeans",
                                                gamma=gamma,
                                                random_state=20)
                labels = clustering.fit_predict(X_split_scaled)

                candidates.append({"labels": labels})
        
        elif self.split_method == "partitioning":
            
            # Compute the MIC values
            mic_metrics = self.compute_mic_metrics(X_cluster, Y_cluster)
            mic_values = {k: v['MIC'] for k, v in mic_metrics.items()}

            # Get the indices of the columns sorted by MIC values in descending order
            mic_values_sorted = sorted(mic_values.items(), key=lambda item: item[1], reverse=True)
            # Get at most two features
            top_features_indices = [index for index, _ in mic_values_sorted[:2]]
            top_features = X_cluster[:, top_features_indices]  # shape: (n_samples, num_top_features)

            # Initialize complete_list_of_features with the selected features
            complete_list_of_features = [top_features[:, i] for i in range(top_features.shape[1])]

            logging.info(f"Running partitioning clustering with top features {top_features_indices}")

            # Compute interaction features (ratio of the two X columns) and the ratios of the X columns and Y)
            if len(top_features_indices) == 2:

                logging.info("Adding ratio X1_X2 feature")

                # Compute ratio of the two X columns
                X1 = top_features[:, 0]
                X2 = top_features[:, 1]
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_X1_X2 = np.true_divide(X1, X2)
                    ratio_X1_X2[~np.isfinite(ratio_X1_X2)] = 0  # Replace infinities and NaNs with zero
                complete_list_of_features.append(ratio_X1_X2)

            # Compute ratios of the X columns and Y
            for i in range(top_features.shape[1]):

                logging.info(f"Adding ratio Y_X{i} feature")

                Xi = top_features[:, i]
                Y = Y_cluster[:, 0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio_Y_Xi = np.true_divide(Y, Xi)
                    ratio_Y_Xi[~np.isfinite(ratio_Y_Xi)] = 0
                complete_list_of_features.append(ratio_Y_Xi)

            for i, split_feature in enumerate(complete_list_of_features):
                # Determine thresholds based on data distribution to bin data uniformly
                num_bins = 10  # Adjust as needed
                thresholds = np.percentile(split_feature,
                                           np.linspace(0, 100, num_bins + 1)[1:-1]  # Exclude 0% and 100%
                                           )
                thresholds = np.unique(thresholds)  # Remove duplicates

                logging.info(f"Partitioning feature {i} with {len(thresholds)} thresholds {thresholds}")

                for threshold in thresholds:
                    # Compute labels
                    labels = (split_feature <= threshold).astype(int)

                    # Ensure that both clusters have at least 5% of the total samples
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    if len(unique_labels) < 2:
                        continue  # Skip this threshold

                    min_cluster_size = 0.01 * n_samples  # 5% of the total samples
                    if np.any(counts < min_cluster_size):
                        continue  # Skip this threshold

                    candidates.append({"labels": labels})

        elif self.split_method == 'density':

            X_split_scaled = StandardScaler().fit_transform(X_split)
            
            # Generate candidate splits by varying eps
            for eps_percentile in [80, 90, 95]:
                try:
                    logging.info(f"Running density clustering with eps_percentile={eps_percentile}")

                    eps = self.estimate_epsilon(X_split_scaled, percentile=eps_percentile)
                    db = DBSCAN(eps=eps, min_samples=5)
                    labels = db.fit_predict(X_split_scaled)
                    unique_labels = set(labels)
                    if len(unique_labels - {-1}) < 2:
                        continue  # Not enough clusters formed
                    # Exclude noise points
                    if -1 in unique_labels:
                        continue  # For simplicity, skip splits with noise
                    # Map labels to 0 and 1
                    labels = np.array([unique_labels.index(label) for label in labels])
                    candidates.append({'labels': labels, 'eps_percentile': eps_percentile})
                except Exception as e:
                    continue  # Skip this candidate if an error occurs

        else:
            raise ValueError("Invalid method. Choose 'spectral', 'partitioning' or 'density'.")

        logging.info(f"Determined {len(candidates)} candidate splits")

        if candidates:
            return candidates
        else:
            return None  # No valid candidates generated
    
    def accept_split(self, score, indicies_left, indicies_right):
        
        return score > 0.05

    def compute_mic_metrics(self, X_cluster, Y_cluster):
        """
        Computes the MIC metrics between X_cluster and Y_cluster using MIC.

        Parameters:
        - X_cluster: ndarray of shape (n_samples_cluster, n_features_X)
            Input variables values in the cluster.
        - Y_cluster: ndarray of shape (n_samples_cluster, 1)
            Output variable values in the cluster.

        Returns:
        - mic_metrics: Dict
            The computed MIC metrics.
        """
        n_samples = X_cluster.shape[0]
        if n_samples < 2:
            return None # Not enough data to compute dependence

        if Y_cluster.shape[1] != 1:
            raise ValueError("Exected Y_cluster to be column vector")

        # Initialize a list to store the absolute correlation coefficients
        n_features_X = X_cluster.shape[1]
        mic_metrics = {}

        # Compute pairwise correlation coefficients between each input and output variable
        for i in range(n_features_X):
            x = X_cluster[:, i]
            y = Y_cluster[:, 0]

            mic_metrics[i] = mic(x, y)

        return mic_metrics

    def compute_cluster_score(self, mic_metrics, weight: float=1.0):
        
        mic_value = mic_metrics[0]["MIC"]
        mev_value = mic_metrics[0]["MEV"]
        mcn_value = mic_metrics[0]["MCN_general"]

        return weight * (mic_value + 1/2 * mev_value)# - 1/2 * mcn_value)
    
    def compute_split_score(self,
                            X_cluster_left, Y_cluster_left,
                            X_cluster_right, Y_cluster_right,
                            mic_metrics_current):

        mic_metrics_left = self.compute_mic_metrics(X_cluster_left, Y_cluster_left)
        mic_metrics_right = self.compute_mic_metrics(X_cluster_right, Y_cluster_right)

        n_samples_left = len(X_cluster_left)
        n_samples_right = len(X_cluster_right)

        score_left = self.compute_cluster_score(mic_metrics_left, weight=n_samples_left/(n_samples_left + n_samples_right))
        score_right = self.compute_cluster_score(mic_metrics_right, weight=n_samples_right/(n_samples_left + n_samples_right))
        score_current = self.compute_cluster_score(mic_metrics_current)

        split_score = np.sum([score_left, score_right]) - score_current

        return split_score, mic_metrics_left, mic_metrics_right

    def construct_similarity_matrix(self, X, gamma=None):
        """
        Constructs a similarity matrix for spectral clustering.

        Parameters:
        - X: ndarray of shape (n_samples_cluster, n_features_X)
            Input variables values in the cluster.
        - gamma: float or None, optional
            Kernel coefficient for the RBF kernel.

        Returns:
        - S: ndarray of shape (n_samples_cluster, n_samples_cluster)
            Similarity matrix.
        """
        if gamma is None:
            # Use default gamma value for RBF kernel
            gamma = 1 / X.shape[1]  # Inverse of number of features
        S = rbf_kernel(X, X, gamma=gamma)
        return S

    def compute_normalized_laplacian(self, S):
        """
        Computes the normalized Laplacian matrix.

        Parameters:
        - S: ndarray of shape (n_samples, n_samples)
            Similarity matrix.

        Returns:
        - L_norm: ndarray of shape (n_samples, n_samples)
            Normalized Laplacian matrix.
        """
        L = laplacian(S, normed=True)
        return L

    def compute_eigenvectors(self, L, k):
        """
        Computes the first k eigenvectors of the Laplacian.

        Parameters:
        - L: ndarray of shape (n_samples, n_samples)
            Laplacian matrix.
        - k: int
            Number of eigenvectors to compute.

        Returns:
        - eigenvectors: ndarray of shape (n_samples, k)
            Matrix of eigenvectors.
        """
        eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
        return eigenvectors

    def estimate_epsilon(self, X_cluster, percentile=90):
        """
        Estimates the epsilon parameter for DBSCAN.

        Parameters:
        - X_cluster: ndarray of shape (n_samples_cluster, n_features_X)
            Input variable values in the cluster.
        - percentile: int, optional
            Percentile to use for estimating epsilon.

        Returns:
        - eps: float
            Estimated epsilon value.
        """
        # Compute the distance to the k-th nearest neighbor (k=5)
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_cluster)
        distances, indices = nbrs.kneighbors(X_cluster)
        distances = np.sort(distances[:, k - 1])
        # Use the specified percentile
        eps = np.percentile(distances, percentile)
        return eps

    def get_clusters(self):
        """
        Returns the list of clusters after fitting.

        Returns:
        - clusters: list of dicts
            Each dict represents a cluster with keys:
            - 'indices': ndarray of indices in X and Y belonging to the cluster.
            - 'depth': int indicating the depth in the hierarchy.
            - 'dependence_measure': float value of the dependence measure.
        """
        return self.clusters
