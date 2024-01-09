import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')


def main():
    logging.info("stratified_kfold.main()")

    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases'
                     '/breast-cancer-wisconsin/wdbc.data',
                     header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    # Create a pipeline
    pipe_lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression()
    )

    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):  # train and test are list of indices
        #logging.debug(f"train = {train};\ntest = {test}")
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print(f'Fold: {k+1:02d}, '
              f'Class distr.: {np.bincount(y_train[train])}, '
              f'Acc.: {score:.3f}')

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f'\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')

    # With cross_val_score
    scores = cross_val_score(estimator=pipe_lr,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             n_jobs=1)
    print(f'CV accuracy scores: {scores}')
    print(f'CV accuracy: {np.mean(scores):.3f} '
          f'+/- {np.std(scores):.3f}')

if __name__ == '__main__':
    main()