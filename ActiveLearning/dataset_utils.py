import pandas
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

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
        message = tokenizer.tokenize(message)
        tokenized_data.append([stemmer.stem(w) for w in message if w not in stop])
    return tokenized_data


def get_padded_sentences(data_set_to_pad, whole_dataset):
    all_words = set([item for sentence in whole_dataset for item in sentence])
    vocab_size = len(all_words)
    longest_sentence = max(len(sentence) for sentence in whole_dataset)

    sentences = join_tokens(data_set_to_pad)
    encoded_sentences = [one_hot(sentence, vocab_size) for sentence in sentences]
    padded_sentences = pad_sequences(encoded_sentences, maxlen=longest_sentence, padding='post')
    return padded_sentences


def join_tokens(data_set):
    joined = []
    for message in data_set:
        joined.append(" ".join([word for word in message]))
    return joined


def convert_class_to_number(encoder, Y):
    encoded_Y = encoder.fit_transform(Y)
    return encoded_Y