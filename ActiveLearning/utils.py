import re

import pandas
import pandas as pd
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np


datasets_path = '../datasets/'

def read_csv(file_name):
    classification_data = pd.read_csv(datasets_path + file_name, encoding="ISO-8859-1", index_col=None)
    return classification_data.to_dict(orient='record')


def write_to_csv(data, filename):
    filename = '../datasets/' + filename

    df = pandas.DataFrame(data)
    df.to_csv(filename, index=False, header=True)


def split_dataset_to_x_y(data_set):
    data_set_x = list([row['message'] for row in data_set])
    data_set_y = list([row['purpose'] for row in data_set])

    return data_set_x, data_set_y


def get_pool_indexes(data_set_y):
    indexes = []
    for i in range(len(data_set_y)):
        if type(data_set_y[i]) is not str:
            indexes.append(i)
    return indexes


def regex_preprocessing(message):
    message = re.sub(r'http\S+', '', message)  # remove all links
    message = re.sub('`{3}.*?`{3}', '', message, flags=re.DOTALL)  # remove all code snippets
    return message


def tokenize(data_set):
    tokenizer = RegexpTokenizer("[a-zA-Z@]+")
    stemmer = SnowballStemmer("english")
    stop = stopwords.words('english')
    excluding = ['against', 'no', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'should', 'shouldn', "shouldn't",
                 "should've" 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
                 'same', 'here', 'likewise', 'above', 'below', 'again', ]
    stop = [word for word in stop if word not in excluding]

    tokenized_data = []
    for message in data_set:
        # message = regex_preprocessing(message)
        message = tokenizer.tokenize(message)
        tokenized_data.append([stemmer.stem(w) for w in message if w not in stop])
    return tokenized_data


def get_padded_sentences(data_set_to_pad, vocab_size, longest_sentence):
    sentences = join_tokens(data_set_to_pad)
    encoded_sentences = [one_hot(sentence, vocab_size) for sentence in sentences]
    padded_sentences = pad_sequences(encoded_sentences, maxlen=longest_sentence, padding='post')
    return padded_sentences


def join_tokens(data_set):
    joined = []
    for message in data_set:
        joined.append(" ".join([word for word in message]))
    return joined


def convert_to_number(encoder, Y):
    return encoder.fit_transform(Y)


def convert_to_binary_vector(encoder, Y):
    encoded_y = convert_to_number(encoder, Y)
    return np_utils.to_categorical(encoded_y)


def get_new_label_from_user():
    while True:
        try:
            entered_value = int(input('LABEL: '))
            if entered_value not in (0, 1, 2, 3, 4):
                raise ValueError
            else:
                return entered_value
        except ValueError:
            print('Incorrect input. Please enter an integer from 0 to 4')


def get_mean_vector(we_model, comment):
    words_in_we = [word for word in comment if word in we_model.vocab]
    if len(words_in_we) >= 1:
        return np.mean(we_model[words_in_we], axis=0)
    else:
        return []


def get_mean_vectors(we, data_set_x, data_set_y):
    indexes_to_remove = []
    data_set_vectors = []
    for index, comment in enumerate(data_set_x):
        vec = get_mean_vector(we, comment)
        if len(vec) > 0:
            data_set_vectors.append(vec)
        else:
            indexes_to_remove.append(index)

    indexes_to_remove.sort(reverse=True)
    for index in indexes_to_remove:
        data_set_y.pop(index)
    return np.asarray(data_set_vectors)
