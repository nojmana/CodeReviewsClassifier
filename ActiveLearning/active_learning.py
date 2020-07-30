from sklearn.ensemble import RandomForestClassifier

import ActiveLearning.dataset_utils
import numpy as np
from sklearn import preprocessing
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

eclipse_file = 'data_eclipse_openj9.csv'
encoder = preprocessing.LabelEncoder()

AL_data_set = ActiveLearning.dataset_utils.read_csv(eclipse_file)

AL_train_set = []
AL_pool = []
for example in AL_data_set:
    if type(example['purpose']) is str:
        AL_train_set.append(example)
    else:
        AL_pool.append(example)

# prepare dataset
AL_train_set_x, AL_train_set_y = ActiveLearning.dataset_utils.split_dataset_to_x_y(AL_train_set)
AL_train_set_x_tokenized = ActiveLearning.dataset_utils.tokenize(AL_train_set_x)
AL_train_set_y_integer = ActiveLearning.dataset_utils.convert_class_to_number(encoder, AL_train_set_y)

AL_pool_x = ActiveLearning.dataset_utils.split_dataset_to_x_y(AL_pool)[0]
AL_pool_x_tokenized = ActiveLearning.dataset_utils.tokenize(AL_pool_x)

AL_train_set_x_padded = ActiveLearning.dataset_utils.get_padded_sentences(AL_train_set_x_tokenized,
                                                                          AL_train_set_x_tokenized + AL_pool_x_tokenized)
AL_pool_x_padded = ActiveLearning.dataset_utils.get_padded_sentences(AL_pool_x_tokenized,
                                                                     AL_train_set_x_tokenized + AL_pool_x_tokenized)

# active learning
queries_number = 10
new_data_set = []

learner = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=uncertainty_sampling,
                        X_training=AL_train_set_x_padded, y_training=AL_train_set_y_integer)

for i in range(queries_number):
    query_idx, query_inst = learner.query(AL_pool_x_padded)

    print('\n\nClasses:')
    for index, encoded_class in enumerate(list(encoder.classes_)):
        print(index, encoded_class)
    print('\n')

    print(AL_pool_x[int(query_idx)])
    new_label = np.array([input()], dtype=int)

    new_data_set.append({'message': AL_pool_x[int(query_idx)], 'purpose': int(new_label)})
    learner.teach(query_inst.reshape(1, -1), new_label)

    AL_pool_x_padded = np.delete(AL_pool_x_padded, query_idx, axis=0)
    AL_pool_x = np.delete(AL_pool_x, query_idx, axis=0)

predictions = learner.predict(AL_pool_x_padded)

print('\n\nPREDICTIONS:\n')
for i, example in enumerate(AL_pool_x):
    print(example, '\npredicted:', predictions[i], '\n')
