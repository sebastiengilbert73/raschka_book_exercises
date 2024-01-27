# Cf. p. 265
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("latent_dirichlet_allocation.main()")

    df = pd.read_csv("./movie_data.csv")
    count = CountVectorizer(stop_words='english', max_df=.1, max_features=500)
    X = count.fit_transform(df['review'].values)

    lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
    X_topics = lda.fit_transform(X)
    logging.info(f"lda.components_.shape = {lda.components_.shape}; type(lda.components_) = {type(lda.components_)}")

    n_top_words = 5
    feature_names = count.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):  # row_ndx, row_as_list
        print(f'Topic {(topic_idx + 1)}:')
        print(' '.join([feature_names[i]
                       for i in topic.argsort()[:-n_top_words - 1: -1] ]))

    horror = X_topics[:, 9].argsort()[::-1]
    for iter_idx, movie_idx in enumerate(horror[:3]):
        print(f'\nHorror movie #{(iter_idx + 1)}:')
        print(df['review'][movie_idx][:300], '...')

if __name__ == '__main__':
    main()