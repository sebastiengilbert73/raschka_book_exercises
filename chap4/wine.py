# Cf. p. 117
import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("wine.main()")

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

    logging.info(f"X_train_std[0: 5, :] = {X_train_std[0: 5, :]}")

    # Logistic regression
    lr = LogisticRegression(penalty='l1',
                            C=1.0,
                            solver='liblinear',
                            multi_class='ovr')
    lr.fit(X_train_std, y_train)
    logging.info(f"Training accuracy: {lr.score(X_train_std, y_train)}")
    logging.info(f"Test accuracy: {lr.score(X_test_std, y_test)}")

    # Plot some coefficients ('weights') as a function of the inverse regularization weights ('params')
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink',
              'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']  # One per coefficient
    weights, params = [], []
    for c in np.arange(-4., 6.):
        lr = LogisticRegression(
            penalty='l1',
            C=10.**c,
            solver='liblinear',
            multi_class='ovr',
            random_state=0
        )
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10.**c)

    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column],
                 label=df_wine.columns[column + 1],
                 color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('Weight coefficient')
    plt.xlabel('C (inverse regularization strength)')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.38, 1.03),
              ncol=1, fancybox=True)
    plt.show()

if __name__ == '__main__':
    main()