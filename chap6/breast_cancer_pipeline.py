# Cf. p. 172
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("breast_cancer_pipeline.main()")

    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                     'machine-learning-databases'
                     '/breast-cancer-wisconsin/wdbc.data',
                     header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    logging.info(f"le.classes_ = {le.classes_}")
    logging.info(f"le.transform(['M', 'B']) = {le.transform(['M', 'B'])}")

    # Train-test split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    # Create a pipeline
    pipe_lr = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression()
    )
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    test_acc = pipe_lr.score(X_test, y_test)
    print(f'Test accuracy: {test_acc:.3f}')

if __name__ == '__main__':
    main()