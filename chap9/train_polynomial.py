# Cf. p. 295
import logging
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("train_polynomial.main()")

    X = np.array([258., 270., 294., 320., 342., 368., 396., 446., 480., 586.])[:, np.newaxis]
    y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368., 391.2, 390.8])
    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)

    lr.fit(X, y)
    X_fit = np.arange(250, 600, 10)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)

    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

    plt.scatter(X, y, label='Training points')
    plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
    plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
    plt.xlabel('Explanatory variable')
    plt.ylabel('Predicted or known target values')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    y_lin_pred = lr.predict(X)
    y_quad_pred = pr.predict(X_quad)
    mse_lin = mean_squared_error(y, y_lin_pred)
    mse_quad = mean_squared_error(y, y_quad_pred)
    logging.info(f"Training MSE linear: {mse_lin:.3f}, quadratic: {mse_quad:.3f}")
    r2_lin = r2_score(y, y_lin_pred)
    r2_quad = r2_score(y, y_quad_pred)
    logging.info(f"Training R^2 linear: {r2_lin:.3f}, quadratic: {r2_quad:.3f}")

    logging.info(f"X_quad = \n{X_quad}")


if __name__ == '__main__':
    main()