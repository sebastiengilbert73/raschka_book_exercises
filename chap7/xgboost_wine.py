# Cf. p. 244
import logging
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("xgboost_wine.main()")

    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                          'machine-learning-databases/'
                          'wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                       'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                       'Proline']
    # Drop 1 class
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4,
                              random_state=1, use_label_encoder=False)
    gbm = model.fit(X_train, y_train)
    y_train_pred = gbm.predict(X_train)
    y_test_pred = gbm.predict(X_test)

    gbm_train = accuracy_score(y_train, y_train_pred)
    gbm_test = accuracy_score(y_test, y_test_pred)
    print(f'XGBoost train/test accuracies '
          f'{gbm_train:.3f}/{gbm_test:.3f}')

if __name__ == '__main__':
    main()