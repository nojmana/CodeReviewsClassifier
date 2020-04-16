import heapq

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


def preprocess_data(dataset):
    tokenizer = RegexpTokenizer("[a-zA-Z@]+")
    stemmer = SnowballStemmer("english")
    stop = stopwords.words('english')
    excluding = ['against', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shouldn', "shouldn't", 'wasn',
                 "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stop = [words for words in stop if words not in excluding]

    for index, message in enumerate(dataset):
        message = tokenizer.tokenize(message)
        dataset[index] = " ".join([stemmer.stem(w) for w in message if w not in stop])
    return dataset


def count_words_frequencies(dataset):
    frequencies = {}
    for message in dataset:
        words = nltk.word_tokenize(message)
        for word in words:
            if word not in frequencies.keys():
                frequencies[word] = 1
            else:
                frequencies[word] += 1
    return frequencies


def create_vector(messages, frequent_words):
    bow_model = []
    for message in messages:
        vector = []
        for word in frequent_words:
            if word in nltk.word_tokenize(message):
                vector.append(1)
            else:
                vector.append(0)
        bow_model.append(vector)
    return np.asarray(bow_model)


def create_bow_model(data_set_x, frequent_words_count):
    data_set_x = preprocess_data(data_set_x)
    word_frequencies = count_words_frequencies(data_set_x)
    frequent_words = heapq.nlargest(frequent_words_count, word_frequencies, key=word_frequencies.get)
    bow_model = create_vector(data_set_x, frequent_words)
    return bow_model
