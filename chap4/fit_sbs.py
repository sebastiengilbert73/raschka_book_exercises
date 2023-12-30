# Cf. p. 131
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sequential_backward_selection import SBS
import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("fit_sbs.main()")


    # Get the standardized dataset
    df_wine = pd.read_csv('https://archive.ics.uci.edu'
                          '/ml/machine-learning-databases/'
                          'wine/wine.data', header=None)
    logging.info(f"df_wine.columns = {df_wine.columns}")
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                       'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                       'Proline']
    logging.info(f"np.unique(df_wine['Class label']) = f{np.unique(df_wine['Class label'])}")
    logging.info(f"df_wine.head = ({df_wine.head()})")

    # Train-test split
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=0.3,
                         random_state=0,
                         stratify=y)

    # Standard scaling
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    # Fit the SBS
    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    # Plot the accuracy graph
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Print the k=3 features
    k3 = list(sbs.subsets_[-3])
    logging.info(f"df_wine.columns[1:][k3] = {df_wine.columns[1:][k3]}")

    # Evaluate on the test dataset
    # With all the features:
    knn.fit(X_train_std, y_train)
    logging.info(f"Training accuracy: {knn.score(X_train_std, y_train)}")
    logging.info(f"Test accuracy: {knn.score(X_test_std, y_test)}")

    # With the selected features
    knn.fit(X_train_std[:, k3], y_train)
    logging.info(f"Training accuracy: {knn.score(X_train_std[:, k3], y_train)}")
    logging.info(f"Test accuracy: {knn.score(X_test_std[:, k3], y_test)}")


if __name__ == '__main__':
    main()