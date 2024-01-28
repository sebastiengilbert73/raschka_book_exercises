# Cf. p. 302
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("fit_ames_housing_random_forest.main()")

    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

    # Convert boolean to 0, 1
    df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
    df = df.dropna(axis=0)

    target = 'SalePrice'
    features = df.columns[df.columns != target]
    X = df[features].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    forest = RandomForestRegressor(
        n_estimators=1000,
        criterion='squared_error',
        random_state=1,
        n_jobs=-1
    )
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    logging.info(f"MAE train: {mae_train:.2f}")
    logging.info(f"MAE test: {mae_test:.2f}")
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    logging.info(f"R^2 train: {r2_train:.2f}")
    logging.info(f"R^2 test: {r2_test:.2f}")

    # Display the residuals
    x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
    x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    ax1.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
    ax2.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
    ax1.set_ylabel('Residuals')
    for ax in (ax1, ax2):
        ax.set_xlabel('Predicted values')
        ax.legend(loc='upper left')
        ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100, color='black', lw=2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()