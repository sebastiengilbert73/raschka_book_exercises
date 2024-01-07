import logging
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from plot_decision_regions_script import plot_decision_regions
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("lda_sklearn.main()")

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

    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)

    lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
    lr = lr.fit(X_train_lda, y_train)
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    X_test_lda = lda.transform(X_test_std)
    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()