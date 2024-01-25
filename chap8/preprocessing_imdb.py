# Cf. p. 248
import logging
import pyprind
import pandas as pd
import os
import sys
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("preprocessing_imdb.main()")

    csv_filepath = "./movie_data.csv"

    if not os.path.exists(csv_filepath):
        basepath = './aclImdb'
        labels = {'pos': 1, 'neg': 0}
        pbar = pyprind.ProgBar(50000, stream=sys.stdout)
        df = pd.DataFrame()
        for s in ('test', 'train'):
            for l in ('pos', 'neg'):
                path = os.path.join(basepath, s, l)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                        txt = infile.read()
                    #df = df.append([[txt, labels[l]]], ignore_index=True)  # deprecated
                    df = pd.concat([df, pd.DataFrame([[txt, labels[l]]])], ignore_index=True)
                    pbar.update()
        df.columns = ['review', 'sentiment']
        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))
        df.to_csv(csv_filepath)
    else:
        df = pd.read_csv(csv_filepath)



if __name__ == '__main__':
    main()