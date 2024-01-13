# Cf. p. 188
import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scipy.stats

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("tuning_hyperparams_via_randomized_search.main()")

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

    pipe_svc = make_pipeline(StandardScaler(),
                             SVC(random_state=1))
    param_range = scipy.stats.loguniform(0.0001, 1000.0)
    param_grid = [{'svc__C': param_range,
                   'svc__kernel': ['linear']},
                  {'svc__C': param_range,
                   'svc__gamma': param_range,
                   'svc__kernel': ['rbf']}]
    rs = RandomizedSearchCV(estimator=pipe_svc,
                      param_distributions=param_grid,
                      scoring='accuracy',
                      cv=10,
                      refit=True,
                      n_jobs=-1,
                      random_state=1,
                      n_iter=20)
    rs = rs.fit(X_train, y_train)
    print(rs.best_score_)
    print(rs.best_params_)

    clf = rs.best_estimator_
    clf.fit(X_train, y_train)  # Not necessary because refit=True
    print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')


if __name__ == '__main__':
    main()