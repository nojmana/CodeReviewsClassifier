from sklearn.linear_model import LogisticRegression

import ActiveLearning.dataset_utils
import numpy as np
from sklearn import preprocessing
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

eclipse_input_file = 'data_eclipse_openj9.csv'
eclipse_output_file = 'data_eclipse_openj9_classified.csv'
encoder = preprocessing.LabelEncoder()

data_set = ActiveLearning.dataset_utils.read_csv(eclipse_input_file)

train_set = []
pool = []
for example in data_set:
    if type(example['purpose']) is str:
        train_set.append(example)
    else:
        pool.append(example)

# prepare dataset
train_set_x, train_set_y = ActiveLearning.dataset_utils.split_dataset_to_x_y(train_set)
train_set_x_tokenized = ActiveLearning.dataset_utils.tokenize(train_set_x)
train_set_y_integer = ActiveLearning.dataset_utils.convert_class_to_number(encoder, train_set_y)

pool_x = ActiveLearning.dataset_utils.split_dataset_to_x_y(pool)[0]
pool_x_tokenized = ActiveLearning.dataset_utils.tokenize(pool_x)

train_set_x_padded = ActiveLearning.dataset_utils.get_padded_sentences(train_set_x_tokenized,
                                                                       train_set_x_tokenized + pool_x_tokenized)
pool_x_padded = ActiveLearning.dataset_utils.get_padded_sentences(pool_x_tokenized,
                                                                  train_set_x_tokenized + pool_x_tokenized)

# active learning
queries_number = 50
new_data_set = []


estimator = LogisticRegression(random_state=40, multi_class='ovr', penalty='elasticnet', l1_ratio=0.1,
                               max_iter=1000000, solver='saga', C=0.1)
learner = ActiveLearner(estimator=estimator, query_strategy=uncertainty_sampling,
                        X_training=train_set_x_padded, y_training=train_set_y_integer)

for i in range(queries_number):
    print('\n\n', i + 1, 'from', queries_number)
    query_idx, query_inst = learner.query(pool_x_padded)

    print('Classes:')
    for index, encoded_class in enumerate(list(encoder.classes_)):
        print(index, encoded_class)
    print('\n')

    print(pool_x[int(query_idx)])
    new_label = np.array([input()], dtype=int)

    new_data_set.append({'message': pool_x[int(query_idx)], 'purpose': int(new_label)})
    learner.teach(query_inst.reshape(1, -1), new_label)

    pool_x_padded = np.delete(pool_x_padded, query_idx, axis=0)
    pool_x = np.delete(pool_x, query_idx, axis=0)

predictions = learner.predict(pool_x_padded)

predicted_set = [{'message': pool_x[i], 'purpose': predictions[i]} for i in range(len(pool_x))]
predicted_set += new_data_set
data_set = ActiveLearning.dataset_utils.write_to_csv(predicted_set, eclipse_output_file)
