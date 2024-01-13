# Cf. p. 190
import logging
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scipy.stats
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef



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
    hs = HalvingRandomSearchCV(estimator=pipe_svc,
                               param_distributions=param_grid,
                               n_candidates='exhaust',
                               resource='n_samples',
                               factor=1.5,
                               random_state=1,
                               n_jobs=-1)
    hs = hs.fit(X_train, y_train)
    print(hs.best_score_)
    print(hs.best_params_)

    clf = hs.best_estimator_
    #clf.fit(X_train, y_train)  # Not necessary because refit=True
    print(f'Test accuracy: {clf.score(X_test, y_test):.3f}')

    y_pred = clf.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    pre_val = precision_score(y_true=y_test, y_pred=y_pred)
    print(f"Precision: {pre_val:.3f}")
    rec_val = recall_score(y_true=y_test, y_pred=y_pred)
    print(f'Recall: {rec_val:.3f}')
    f1_val = f1_score(y_true=y_test, y_pred=y_pred)
    print(f'F1: {f1_val:.3f}')
    mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)
    print(f'MCC: {mcc_val:.3f}')


if __name__ == '__main__':
    main()