from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from Classifier.grid_display import GridSearch_table_plot


def measure_efficiency(y, predictions):
    print('Accuracy Score : ', str(accuracy_score(y, predictions)))


class Classifier:
    def __init__(self, seed, train_bow_model, train_set_y, test_bow_model, test_set_y):
        self.seed = seed
        self.train_bow_model = train_bow_model
        self.train_set_y = train_set_y
        self.test_bow_model = test_bow_model
        self.test_set_y = test_set_y

    def logistic_regression(self):
        log_reg = LogisticRegression(random_state=self.seed, max_iter=1000000, solver='liblinear', C=0.1)
        log_reg.fit(self.train_bow_model, self.train_set_y)
        predictions = log_reg.predict(self.test_bow_model)
        measure_efficiency(self.test_set_y, predictions)

    def grid_search_logistic_regression(self):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'solver': ['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs']}
        grid_search = GridSearchCV(LogisticRegression(random_state=self.seed, max_iter=1000000), param_grid, verbose=1)
        grid_search.fit(self.train_bow_model, self.train_set_y)
        predictions = grid_search.predict(self.test_bow_model)
        measure_efficiency(self.test_set_y, predictions)
        GridSearch_table_plot(grid_search, "C", graph=False, negative=False)

    def random_forest(self, estimators_number):
        rfc = RandomForestClassifier(estimators_number, random_state=self.seed)
        rfc.fit(self.train_bow_model, self.train_set_y)
        predictions = rfc.predict(self.test_bow_model)
        measure_efficiency(self.test_set_y, predictions)

