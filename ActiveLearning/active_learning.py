import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from gensim.models.keyedvectors import KeyedVectors

from ActiveLearning import utils


eclipse_input_file = 'data_eclipse_openj9.csv'
eclipse_output_file = 'data_eclipse_openj9_classified.csv'
we_path = '../Classifier/SO_vectors_200.bin'
encoder = preprocessing.LabelEncoder()


def print_classes():
    print('Classes:')
    for index, encoded_class in enumerate(list(encoder.classes_)):
        print(index, encoded_class)
    print('\n')


print("Loading word embeddings...")
we = KeyedVectors.load_word2vec_format(we_path, binary=True)
print("Loaded word embeddings!")


data_set = utils.read_csv(eclipse_input_file)
train_set = []
pool = []
for example in data_set:
    if type(example['purpose']) is str:
        train_set.append(example)
    else:
        pool.append(example)

train_set_x, train_set_y = utils.split_dataset_to_x_y(train_set)
train_set_x_tokenized = utils.tokenize(train_set_x)
train_set_x_mean_vectors = utils.get_mean_vectors(we, train_set_x_tokenized, train_set_y)

pool_x = utils.split_dataset_to_x_y(pool)[0]
pool_x_tokenized = utils.tokenize(pool_x)
pool_x_mean_vectors = utils.get_mean_vectors(we, pool_x_tokenized, pool_x)


# active learning
encoder.fit(train_set_y)
queries_number = 50
new_data_set = []

classifier = LogisticRegression(random_state=40, multi_class='ovr', penalty='elasticnet', l1_ratio=0.1,
                                max_iter=1000000, solver='saga', C=0.1)
learner = ActiveLearner(estimator=classifier, query_strategy=entropy_sampling,
                        X_training=train_set_x_mean_vectors, y_training=np.asarray(train_set_y))

for i in range(queries_number):
    print('\n\n', i + 1, 'from', queries_number)
    print_classes()

    query_idx, query_inst = learner.query(pool_x_mean_vectors)
    message = pool_x[int(query_idx)]
    print('MESSAGE:', utils.regex_preprocessing(message))

    new_label = np.array([utils.get_new_label_from_user()], dtype=int)
    new_data_set.append({'message': pool_x[int(query_idx)],
                         'purpose': encoder.inverse_transform(new_label)[0]})
    learner.teach(query_inst, new_label)

    pool_x_mean_vectors = np.delete(pool_x_mean_vectors, query_idx, axis=0)
    pool_x = np.delete(pool_x, query_idx, axis=0)

predictions = learner.predict(pool_x_mean_vectors)

predicted_set = [{'message': pool_x[i], 'purpose': predictions[i]} for i in range(len(pool_x))]
predicted_set += new_data_set
data_set = utils.write_to_csv(predicted_set, eclipse_output_file)
