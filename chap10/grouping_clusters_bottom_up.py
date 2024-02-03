# Cf. p. 321
import logging
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("grouping_clusters_bottom_up.main")

    np.random.seed(123)
    variables =['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
    X = np.random.random_sample([5, 3]) *10  # (5, 3) in the range [0, 10]
    df = pd.DataFrame(X, columns=variables, index=labels)
    logging.info(f"df = \n{df}")

    row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
    # squareform() makes a redundant square matrix from the condensed pair-wise distances vector returned
    # by pdist()
    logging.info(f"row_dist =\n{row_dist}")

    row_clusters = linkage(df.values, method='complete', metric='euclidean')
    row_clusters_df = pd.DataFrame(
        row_clusters,
        columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
        index=[f'cluster {(i + 1)}' for i in range(row_clusters.shape[0])]
    )
    logging.info(f"row_clusters_df =\n{row_clusters_df}")

    # Display the linkage matrix as a dendrogram
    row_dendr = dendrogram(
        row_clusters,
        labels=labels,
    )
    plt.tight_layout()
    plt.show()

    # Attach a dendrogram to a heatmap
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    row_dendr = dendrogram(
        row_clusters,
        orientation='left'
    )
    df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
    axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    axd.set_xticks([])
    axd.set_yticks([])
    for i in axd.spines.values():
        i.set_visible(False)
    fig.colorbar(cax)
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))
    plt.show()

    # Agglomerative clustering
    # Cf. p. 328
    ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
    labels = ac.fit_predict(X)
    logging.info(f"labels = {labels}")

if __name__ == '__main__':
    main()