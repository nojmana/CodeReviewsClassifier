from Classifier import input_reader, bag_of_words, logistic_regression

gerrit_file = 'gerrit-wireshark-train-test-v4.xlsx'
seed = 40


if __name__ == "__main__":
    data_set = input_reader.read_excel(gerrit_file, 'train')
    train_set, test_set = input_reader.split_set(data_set, 0.9, seed)

    train_set_x = list([row['message'] for row in train_set])
    train_set_y = list([row['purpose'] for row in train_set])
    test_set_x = list([row['message'] for row in test_set])
    test_set_y = list([row['purpose'] for row in test_set])

    frequent_words_count = 100
    train_bow_model = bag_of_words.create_bow_model(train_set_x, frequent_words_count)
    test_bow_model = bag_of_words.create_bow_model(test_set_x, frequent_words_count)

    log_reg = logistic_regression.LogisticRegressionImpl(seed)
    log_reg.train(train_bow_model, train_set_y)
    predictions = log_reg.predict(test_bow_model)
    log_reg.measure_efficiency(test_bow_model, test_set_y)
