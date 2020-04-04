import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def preprocess_data(dataset):
    tokenizer = RegexpTokenizer("[a-zA-Z@]+")
    stemmer = SnowballStemmer("english")
    stop = stopwords.words('english')
    excluding = ['against', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shouldn', "shouldn't", 'wasn',
                 "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stop = [words for words in stop if words not in excluding]

    for row in dataset:
        message = row['message']
        message = tokenizer.tokenize(message)
        message = [stemmer.stem(w) for w in message if w not in stop]
        row['message'] = " ".join(message)


def create_bow(dataset):
    bow = {}
    for row in dataset:
        words = nltk.word_tokenize(row['message'])
        for word in words:
            if word not in bow.keys():
                bow[word] = 1
            else:
                bow[word] += 1
    return bow

