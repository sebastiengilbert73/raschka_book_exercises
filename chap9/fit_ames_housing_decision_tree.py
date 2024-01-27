# Cf. p. 300
import logging
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)

def main():
    logging.info("fit_ames_housing_decision_tree.main()")

    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)

    # Convert boolean to 0, 1
    df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
    df = df.dropna(axis=0)

    X = df[['Gr Liv Area']].values
    y = df['SalePrice'].values

    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X, y)

    sort_idx = X.flatten().argsort()
    lin_regplot(X[sort_idx], y[sort_idx], tree)
    plt.xlabel('Living area above ground in square feet')
    plt.ylabel('Sale price in U.S. dollars')
    plt.show()


if __name__ == '__main__':
    main()