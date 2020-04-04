from Classifier import input_reader, bag_of_words

gerrit_file = 'gerrit-wireshark-train-test-v4.xlsx'

if __name__ == "__main__":
    data_set = input_reader.read_excel(gerrit_file, 'train')
    split_index = int(0.9*len(data_set))
    train_set, test_set = data_set[:split_index], data_set[split_index:]
    bag_of_words = bag_of_words
    bag_of_words.preprocess_data(train_set)
    bow = bag_of_words.create_bow(train_set)
