# Cf. p. 142
import logging
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("extracting_pca.main()")

    df_wine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        header=None
    )

    # Train-test split
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3,
                         stratify=y,
                         random_state=0)
    # Standardize the features
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # Computation of the covariance matrix
    import numpy as np
    cov_mat = np.cov(X_train_std.T)

    # Eigendecomposition
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    logging.info(f"eigen_vals = {eigen_vals}")

    # Plot the explained variance
    tot = sum(eigen_vals)
    var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    import matplotlib.pyplot as plt
    plt.bar(range(1, 14), var_exp, align='center',
            label='Individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Feature transformation
    # Inverse sorting of the eigenvalues
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    # Keep the two highest pairs
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    logging.info(f"w = {w}")

    # Transform the features
    logging.info(f"X_train_std[0].dot(w) = {X_train_std[0].dot(w)}")
    X_train_pca = X_train_std.dot(w)
    colors = ['r', 'g', 'b']
    markers = ['o', 's', '^']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==l, 0],
                    X_train_pca[y_train==l, 1],
                    c=c, label=f'Class {l}', marker=m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()