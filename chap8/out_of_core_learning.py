# Cf. p. 261
import logging
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # Skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def main():
    logging.info(f"out_of_core_learning.main()")


if __name__ == '__main__':
    main()