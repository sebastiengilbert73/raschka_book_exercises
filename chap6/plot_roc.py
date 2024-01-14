# Cf. p. 198
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info(f"plot_roc.main()")

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

    pipe_lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(penalty='l2', random_state=1, solver='lbfgs', C=100.0)
    )
    X_train2 = X_train[:, [4, 14]]
    cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))

    fig = plt.figure(figsize=(7, 5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train, test) in enumerate(cv):
        probas = pipe_lr.fit(
            X_train2[train],
            y_train[train],
        ).predict_proba(X_train2[test])
        fpr, tpr, thresholds = roc_curve(y_train[test],
                                         probas[:, 1],
                                         pos_label=1)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,
                 tpr,
                 label=f'ROC fold {i+1} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6),
             label='Random guessing (area=0.5)')
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label=f'Mean ROC (area={mean_auc:.2f})', lw=2)
    plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='Perfect performance (area=1.0)')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()