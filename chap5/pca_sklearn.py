# Cf. p. 150
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from plot_decision_regions_script import plot_decision_regions
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("pca_sklearn.main()")

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

    # Initialize the PCA transformer and logistic regression estimator
    pca = PCA(n_components=2)
    lr = LogisticRegression(multi_class='ovr',
                            random_state=1,
                            solver='lbfgs')
    # Dimensionality reduction
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # Fitting the logistic regression model on the reduced dataset:
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    # Plot the test data
    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    # Explained variance ratio
    pca = PCA(n_components=None)
    X_train_pca = pca.fit_transform(X_train_std)
    logging.info(f"pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}")

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    fig, ax = plt.subplots()
    ax.bar(range(13), loadings[:, 0], align='center')
    ax.set_ylabel('Loadings for PC1')
    ax.set_xticks(range(13))
    ax.set_xticklabels(df_wine.columns[1:], rotation=90)
    plt.ylim([-1, 1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()