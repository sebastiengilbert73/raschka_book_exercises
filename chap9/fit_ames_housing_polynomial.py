# Cf. p. 297
import logging
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("fit_ames_housing_polynomial.main()")

    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

    # Convert boolean to 0, 1
    df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
    df = df.dropna(axis=0)

    X = df[['Overall Qual']].values
    y = df['SalePrice'].values
    # Remove outliers
    X = X[(df['Gr Liv Area'] < 4000)]
    y = y[(df['Gr Liv Area'] < 4000)]

    regr = LinearRegression()
    # Create quadratic and cubic features
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)
    logging.info(f"X.shape = {X.shape}; X_quad.shape = {X_quad.shape}; X_cubic.shape = {X_cubic.shape}")

    # Fit to features
    X_fit = np.arange(X.min() - 1, X.max() + 2, 1)[:, np.newaxis]
    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))

    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))

    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))

    # Plot results
    plt.scatter(X, y, label='Training points', color='lightgray')
    plt.plot(X_fit, y_lin_fit, label=f'Linear (d=1), $R^2$={linear_r2:.2f}', color='blue', lw=2, linestyle=':')
    plt.plot(X_fit, y_quad_fit, label=f'Quadratic (d=2), $R^2$={quadratic_r2:.2f}', color='red', lw=2, linestyle='-')
    plt.plot(X_fit, y_cubic_fit, label=f'Cubic (d=3), $R^2$={cubic_r2:.2f}', color='green', lw=2, linestyle='--')
    plt.xlabel('Overall quality of the house')
    plt.ylabel('Sale price in U.S. dollars')
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':
    main()