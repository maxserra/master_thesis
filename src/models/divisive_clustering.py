import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

class DivisiveHierarchicalClustering:
    """
    A class to perform divisive hierarchical clustering on data with one input and one output variable.
    """

    def __init__(self, X, Y, min_cluster_size=100, split_method='spectral', max_depth=None):
        """
        Initializes the clustering algorithm with data and parameters.

        Parameters:
        - X: ndarray of shape (n_samples,)
            Input variable values.
        - Y: ndarray of shape (n_samples,)
            Output variable values.
        - min_cluster_size: int, optional
            Minimum number of samples in a cluster to consider splitting further.
        - split_method: str, optional
            Clustering method for splitting ('spectral' or 'density').
        - max_depth: int or None, optional
            Maximum depth of the clustering hierarchy.
        """
        self.X = X
        self.Y = Y
        self.min_cluster_size = min_cluster_size
        self.split_method = split_method
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
            'dependence_measure': self.compute_dependence_measure(self.X[initial_indices], self.Y[initial_indices])
        }
        queue = [initial_cluster]

        while queue:
            current_cluster = queue.pop(0)
            indices = current_cluster['indices']
            depth = current_cluster['depth']

            # Check stopping criteria
            if len(indices) <= self.min_cluster_size or (self.max_depth is not None and depth >= self.max_depth):
                self.clusters.append(current_cluster)
                continue

            # Attempt to split the cluster
            split_result = self.split_cluster(self.X[indices], self.Y[indices])

            if split_result is not None:
                labels = split_result['labels']
                # Split indices based on labels
                indices_left = indices[labels == 0]
                indices_right = indices[labels == 1]

                # Compute dependence measures
                dep_left = self.compute_dependence_measure(self.X[indices_left], self.Y[indices_left])
                dep_right = self.compute_dependence_measure(self.X[indices_right], self.Y[indices_right])
                dep_current = current_cluster['dependence_measure']

                # Accept split if dependence measure improves
                if self.accept_split(dep_left, dep_right, dep_current):
                    # Add child clusters to the queue
                    queue.append({
                        'indices': indices_left,
                        'depth': depth + 1,
                        'dependence_measure': dep_left
                    })
                    queue.append({
                        'indices': indices_right,
                        'depth': depth + 1,
                        'dependence_measure': dep_right
                    })
                    continue  # Proceed to next cluster

            # If split not accepted, add current cluster to final clusters
            self.clusters.append(current_cluster)

    def split_cluster(self, X_cluster, Y_cluster):
        """
        Attempts to split a given cluster using the specified clustering method.

        Parameters:
        - X_cluster: ndarray of shape (n_samples_cluster,)
            Input variable values in the cluster.
        - Y_cluster: ndarray of shape (n_samples_cluster,)
            Output variable values in the cluster.

        Returns:
        - result: dict or None
            If successful, returns dict with key 'labels' (ndarray of cluster labels).
            Returns None if the cluster cannot be split.
        """
        n_samples = X_cluster.shape[0]
        if n_samples < 2:
            return None  # Not enough points to split

        if self.split_method == 'spectral':
            # Construct similarity matrix
            S = self.construct_similarity_matrix(np.hstack([X_cluster.reshape(-1, 1), Y_cluster.reshape(-1, 1)]))
            # Compute Laplacian
            L = self.compute_normalized_laplacian(S)
            # Compute eigenvectors
            try:
                eigenvectors = self.compute_eigenvectors(L, k=2)
            except Exception as e:
                print(f"Eigenvalue computation failed: {e}")
                return None
            # Cluster using k-means on the first eigenvector
            labels = KMeans(n_clusters=2, random_state=42).fit_predict(eigenvectors)
        elif self.split_method == 'density':
            # Estimate epsilon
            eps = self.estimate_epsilon(X_cluster)
            # Perform DBSCAN
            db = DBSCAN(eps=eps, min_samples=5)
            labels = db.fit_predict(X_cluster.reshape(-1, 1))
            # Check if at least two clusters are formed
            unique_labels = set(labels)
            if len(unique_labels - {-1}) < 2:
                return None  # Cannot split
            # Relabel clusters to 0 and 1, ignore noise (-1)
            labels = np.array([label if label != -1 else -1 for label in labels])
            # Exclude noise points
            if np.any(labels == -1):
                # Optionally, handle noise points here
                return None  # For simplicity, we do not split if noise is present
            # Map labels to 0 and 1
            unique_labels = list(unique_labels)
            labels = np.array([unique_labels.index(label) for label in labels])
        else:
            raise ValueError("Invalid method. Choose 'spectral' or 'density'.")

        return {'labels': labels}
    
    def accept_split(self, value_left, value_right, value_current):
        
        # if 2 * np.mean([value_left, value_right]) > value_current:
        if value_left + value_right > value_current:
            return True
        
        return False

    def compute_dependence_measure(self, X_cluster, Y_cluster):
        """
        Computes the dependence measure (e.g., correlation coefficient) for a cluster.

        Parameters:
        - X_cluster: ndarray of shape (n_samples_cluster,)
            Input variable values in the cluster.
        - Y_cluster: ndarray of shape (n_samples_cluster,)
            Output variable values in the cluster.

        Returns:
        - dependence_measure: float
            The computed dependence measure.
        """
        if len(X_cluster) < 2:
            return 0.0  # Not enough data to compute correlation
        correlation = np.corrcoef(X_cluster, Y_cluster)[0, 1]
        return abs(correlation)  # Absolute value to measure strength

    def construct_similarity_matrix(self, X, sigma=None):
        """
        Constructs a similarity matrix for spectral clustering.

        Parameters:
        - X: ndarray of shape (n_samples_cluster,)
            Input variable values in the cluster.
        - sigma: float or None, optional
            Bandwidth parameter for the RBF kernel.

        Returns:
        - S: ndarray of shape (n_samples_cluster, n_samples_cluster)
            Similarity matrix.
        """
        if sigma is None:
            sigma = np.std(X) or 1.0
        # X = X.reshape(-1, 1)
        S = rbf_kernel(X, X, gamma=1 / (2 * sigma ** 2))
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

    def estimate_epsilon(self, X_cluster):
        """
        Estimates the epsilon parameter for DBSCAN.

        Parameters:
        - X_cluster: ndarray of shape (n_samples_cluster,)
            Input variable values in the cluster.

        Returns:
        - eps: float
            Estimated epsilon value.
        """
        # Compute the distance to the k-th nearest neighbor (k=5)
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_cluster.reshape(-1, 1))
        distances, indices = nbrs.kneighbors(X_cluster.reshape(-1, 1))
        distances = np.sort(distances[:, k - 1])
        # Use the elbow method or a percentile
        eps = np.percentile(distances, 90)
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
