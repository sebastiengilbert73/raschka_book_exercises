# Cf. p. 272
import logging
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("exploring_ames_housing.main()")

    columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area', 'Central Air', 'Total Bsmt SF', 'SalePrice']
    df = pd.read_csv('http://jse.amstat.org/v19n3/decock/AmesHousing.txt', sep='\t', usecols=columns)
    print(df.head())
    logging.info(f"df.shape = {df.shape}")

    # Convert boolean to 0, 1
    df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
    logging.info(f"df.isnull().sum() = \n{df.isnull().sum()}")
    df = df.dropna(axis=0)
    logging.info(f"df.isnull().sum() = \n{df.isnull().sum()}")

    # Visualize pair-wise correlations
    scatterplotmatrix(df.values, figsize=(12, 10), names=df.columns, alpha=0.5)
    cm = np.corrcoef(df.values.T)
    hm = heatmap(cm, row_names=df.columns, column_names=df.columns)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()