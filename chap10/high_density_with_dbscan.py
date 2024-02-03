# Cf. p. 329
import logging
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("high_density_with_dbscan.main()")

    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    # Test k-means
    km = KMeans(n_clusters=2, random_state=0)
    y_km = km.fit_predict(X)
    ax1.scatter(X[y_km==0, 0], X[y_km==0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='cluster 1')
    ax1.scatter(X[y_km==1, 0], X[y_km==1, 1], c='red', edgecolor='black', marker='s', s=40, label='cluster 2')
    ax1.set_title('K-means clustering')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')

    # Test Agglomerative clustering
    ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
    y_ac = ac.fit_predict(X)
    ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='cluster 1')
    ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red', edgecolor='black', marker='s', s=40, label='cluster 2')
    ax1.set_title('Agglomerative clustering')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')

    # DBSCAN
    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(X)
    ax3.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='cluster 1')
    ax3.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c='red', edgecolor='black', marker='s', s=40, label='cluster 2')
    ax3.set_title('DBSCAN')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')


    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()