# Cf. p. 156
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("computing_lda.main()")

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

    # Compute the mean vectors
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in range(1, 4):
        mean_vecs.append(np.mean(
            X_train_std[y_train==label], axis=0
        ))
        print(f'MV {label}: {mean_vecs[label - 1]}\n')

    # Within-class scatter matrices
    d = 13  # Number of features
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 4), mean_vecs):
        """class_scatter = np.zeros((d, d))
        for row in X_train_std[y_train==label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)
            class_scatter += (row - mv).dot((row - mv).T)
        """
        # Cf. p. 158
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter
    print('Within-class scatter matrix: '
          f'{S_W.shape[0]}x{S_W.shape[1]}')

    # Between-class scatter matrix
    mean_overall = np.mean(X_train_std, axis=0)
    mean_overall = mean_overall.reshape(d, 1)

    S_B = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X_train_std[y_train==i + 1, :].shape[0]  # Number of observations of class i + 1
        mean_vec = mean_vec.reshape(d, 1)  # Make column vector
        S_B += n * (mean_vec - mean_overall).dot(
            (mean_vec - mean_overall).T
        )
    print('Between-class scatter matrix: '
          f'{S_B.shape[0]}x{S_B.shape[1]}')

    # Eigendecomposition
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                   for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in descending order:\n')
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

    # Discriminability
    tot = sum(eigen_vals.real)
    discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)
    plt.bar(range(1, 14), discr, align='center', label="Individual discriminability")
    plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative discriminability')
    plt.ylabel('"Discriminability" ratio')
    plt.xlabel('Linear Discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Transformation matrix
    w = np.stack((eigen_pairs[0][1][:, np.newaxis].real,
                  eigen_pairs[1][1][:, np.newaxis].real))
    print('Matrix W:\n', w)

    # Projection
    X_train_lda = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['o', 's', '^']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train==l, 0],
                    X_train_lda[y_train==l, 1] * (-1),
                    c=c, label=f'Class {l}', marker=m)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()