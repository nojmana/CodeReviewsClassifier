from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


class LogisticRegressionImpl:
    def __init__(self, seed):
        self.log_reg = LogisticRegression(random_state=seed, max_iter=200)

    def train(self, x, y):
        self.log_reg.fit(x, y)

    def predict(self, x):
        return self.log_reg.predict(x)

    def measure_efficiency(self, y, predictions):
        print('Accuracy Score : ', str(accuracy_score(y, predictions)))

    def grid_search(self):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      'solver': ['liblinear']}
        return GridSearchCV(self.log_reg, param_grid, verbose=1)
