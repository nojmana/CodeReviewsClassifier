import heapq

from Classifier import input_reader, bag_of_words

gerrit_file = 'gerrit-wireshark-train-test-v4.xlsx'

if __name__ == "__main__":
    data_set = input_reader.read_excel(gerrit_file, 'train')
    split_index = int(0.9*len(data_set))

    train_set, test_set = data_set[:split_index], data_set[split_index:]
    train_set_x = list([row['message'] for row in train_set])
    train_set_y = list([row['purpose'] for row in train_set])

    bag_of_words.preprocess_data(train_set_x)
    word_frequencies = bag_of_words.count_words_frequencies(train_set_x)
    train_frequent_words = heapq.nlargest(100, word_frequencies, key=word_frequencies.get)
    train_bow_model = bag_of_words.create_bow_model(train_set_x, train_frequent_words)
