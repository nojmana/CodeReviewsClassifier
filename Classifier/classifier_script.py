from Classifier import input_reader, bag_of_words, logistic_regression
from Classifier.grid_display import GridSearch_table_plot

gerrit_file = 'gerrit-wireshark-train-test-v4.xlsx'
seed = 40


if __name__ == "__main__":
    data_set = input_reader.read_excel(gerrit_file, 'train')

    data_set_x = list([row['message'] for row in data_set])
    data_set_y = list([row['purpose'] for row in data_set])
    data_set_y = input_reader.encode_labels(data_set_y)

    train_set_x, test_set_x = input_reader.split_set(data_set_x, 0.9, seed)
    train_set_y, test_set_y = input_reader.split_set(data_set_y, 0.9, seed)

    frequent_words_count = 100
    train_bow_model = bag_of_words.create_bow_model(train_set_x, frequent_words_count)
    test_bow_model = bag_of_words.create_bow_model(test_set_x, frequent_words_count)

    log_reg = logistic_regression.LogisticRegressionImpl(seed)
    grid_search = log_reg.grid_search()
    grid_search.fit(train_bow_model, train_set_y)
    predictions = grid_search.predict(test_bow_model)
    log_reg.measure_efficiency(test_set_y, predictions)

    GridSearch_table_plot(grid_search, "C", graph=False, negative=False)
