# Cf. p. 250
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text

def tokenizer(text):
    return text.split()



def main():
    logging.info("intro_bagofwords.main()")

    count = CountVectorizer()
    docs = np.array(['The sun is shining',
                     'The weather is sweet',
                     'The sun is shining, the weather is sweert, '
                     'and one and one is two'])
    bag = count.fit_transform(docs)
    logging.info(f"count.vocabulary_ = {count.vocabulary_}")
    logging.info(f"bag.toarray() = \n{bag.toarray()}")

    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    np.set_printoptions(precision=2)
    print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

    df = pd.read_csv("./movie_data.csv")
    df['review'] = df['review'].apply(preprocessor)

    porter = PorterStemmer()

    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    stop = stopwords.words('english')

    X_train = df.loc[: 25000, 'review'].values
    y_train = df.loc[: 25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values

    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)
    small_param_grid = [
        {
            'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [None],
            'vect__tokenizer': [tokenizer, tokenizer_porter],
            'clf__penalty': ['l2'],
            'clf__C': [1.0, 10.0]
        },
        {
            'vect__ngram_range': [(1, 1)],
            'vect__stop_words': [stop, None],
            'vect__tokenizer': [tokenizer],
            'vect__use_idf': [False],
            'vect__norm': [None],
            'clf__penalty': ['l2'],
            'clf__C': [1.0, 10.0]
        }
    ]
    lr_tfidf = Pipeline([
        ('vect', tfidf),
        ('clf', LogisticRegression(solver='liblinear'))
    ])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=1)
    gs_lr_tfidf.fit(X_train, y_train)

    # Print the best parameters
    logging.info(f"Best parameter set: {gs_lr_tfidf.best_params_}")
    logging.info(f"CV Accuracy: {gs_lr_tfidf.best_score_:.3f}")
    clf = gs_lr_tfidf.best_estimator_
    logging.info(f"Test accuracy:  {clf.score(X_test, y_test):.3f}")

if __name__ == '__main__':
    main()