# Cf. p. 283
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)

def main():
    logging.info("train_sklearn_ransac.main()")

    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

    # Convert boolean to 0, 1
    df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
    df = df.dropna(axis=0)

    X = df[['Gr Liv Area']].values
    y = df['SalePrice'].values

    ransac = RANSACRegressor(
        LinearRegression(),
        max_trials=100,
        min_samples=0.95,
        residual_threshold=None,  # Automatically chosen by MAD estimate
        random_state=123
    )
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(ransac.inlier_mask_)
    line_X = np.arange(0, 5000, 1000)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white', marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='black', lw=2)
    plt.xlabel('Living area above ground in square feet')
    plt.ylabel('Sale price in U.S. dollars')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    logging.info(f"Slope: {ransac.estimator_.coef_[0]:.3f}")
    logging.info(f"Intercept: {ransac.estimator_.intercept_:.3f}")

if __name__ == '__main__':
    main()