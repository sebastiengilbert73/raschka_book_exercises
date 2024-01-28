# Cf. p.306
import logging
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("make_blobs.main()")

    X, y = make_blobs(
        n_samples=150,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        shuffle=True,
        random_state=0
    )
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()
    plt.tight_layout()
    plt.show()

    logging.info(f"y =\n{y}")

if __name__ == '__main__':
    main()