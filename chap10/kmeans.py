# Cf. p.308
import logging
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("kmeans.main()")

    X, y = make_blobs(
        n_samples=150,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        shuffle=True,
        random_state=0
    )
    """plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()
    plt.tight_layout()
    plt.show()
    """
    km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)

    # Display the clusters
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Cluster 1')
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Cluster 2')
    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black', label='Cluster 3')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centroids')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()