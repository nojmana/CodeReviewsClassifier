from sklearn.linear_model import LogisticRegression


class LogisticRegressionImpl:
    def __init__(self, seed):
        self.log_reg = LogisticRegression(random_state=seed)

    def train(self, x, y):
        self.log_reg.fit(x, y)

    def predict(self, x):
        return self.log_reg.predict(x)

    def measure_efficiency(self, x, y):
        score = self.log_reg.score(x, y)
        print('Achieved score: ', score)

