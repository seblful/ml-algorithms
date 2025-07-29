import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GaussianNB:
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self.classes = np.unique(y)
        num_classes, num_feautures = len(self.classes), X.shape[1]

        self.means = np.zeros((num_classes, num_feautures))
        self.variances = np.zeros((num_classes, num_feautures))
        self.priors = np.zeros(num_classes)

        for i, class_ in enumerate(self.classes):
            Xk = X[y == class_]
            self.means[i] = Xk.mean(axis=0)
            self.variances[i] = Xk.var(axis=0)

            self.priors[i] = Xk.shape[0] / X.shape[0]

        self.fitted = True

    def log_gaussian(self, X: np.array):
        num = -0.5 * (X[:, None, :] - self.means) ** 2 / self.variances
        pi = -0.5 * np.log(2 * np.pi * self.variances)
        log_prob = pi + num
        log_prob_sum = log_prob.sum(axis=2)

        return log_prob_sum

    def predict(self, X: np.array):
        if self.fitted is False:
            raise ValueError("Method should be fitted first.")

        log_prior = np.log(self.priors)
        log_likelihood = self.log_gaussian(X)

        probs = log_prior + log_likelihood
        argmax = np.argmax(probs, axis=1)
        result = self.classes[argmax]

        return result


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    print("Score:", score)


if __name__ == "__main__":
    main()
