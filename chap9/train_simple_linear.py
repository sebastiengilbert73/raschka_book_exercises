# Cf. p. 280
import logging
from linear_regression import LinearRegressionGD
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)

def main():
    logging.info("train_simple_linear.main()")

    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

    # Convert boolean to 0, 1
    df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
    df = df.dropna(axis=0)

    X = df[['Gr Liv Area']].values
    y = df['SalePrice'].values

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
    lr = LinearRegressionGD(eta=0.1)
    lr.fit(X_std, y_std)

    plt.plot(range(1, lr.n_iter + 1), lr.losses_)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.show()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Living area above ground (standardized)')
    plt.ylabel('Sale price (standardized)')
    plt.show()

    feature_std = sc_x.transform(np.array([[2500]]))
    target_std = lr.predict(feature_std)
    target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
    logging.info(f"Sale price: ${target_reverted.flatten()[0]:.2f}")


if __name__ == '__main__':
    main()