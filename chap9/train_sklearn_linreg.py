# Cf. p. 283
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)

def main():
    logging.info("train_sklearn_linreg.main()")

    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

    # Convert boolean to 0, 1
    df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
    df = df.dropna(axis=0)

    X = df[['Gr Liv Area']].values
    y = df['SalePrice'].values

    slr = LinearRegression()
    slr.fit(X, y)
    y_pred = slr.predict(X)
    logging.info(f"Slope: {slr.coef_[0]:.3f}")
    logging.info(f"Intercept: {slr.intercept_:.3f}")

    lin_regplot(X, y, slr)
    plt.xlabel('Living area above ground in square feet')
    plt.ylabel('Sale price in U.S. dollars')
    plt.tight_layout()
    plt.show()

    # Analytical solution
    # Adding a column vector of ones
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))
    logging.info(f"Slope: {w[1]:.3f}")
    logging.info(f"Intercept: {w[0]:.3f}")

if __name__ == '__main__':
    main()