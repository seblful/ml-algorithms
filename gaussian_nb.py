import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GaussianNB:
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)
        num_classes, num_feautures = len(self.classes), X.shape[1]

        self.means = np.zeros((num_classes, num_feautures))
        self.variances = np.zeros((num_classes, num_feautures))
        self.priors = np.zeros(num_classes)

        for i, class_ in enumerate(self.classes):
            Xk = X[y == class_]
            self.means[i] = Xk.mean(axis=0)
            self.variances[i] = Xk.var(axis=0, ddof=1)

            self.priors[i] = Xk.shape[0] / X.shape[0]

        self.fitted = True

    def log_gaussian(
        self,
        X: np.array,
        prior: np.array,
        means: np.array,
        variances: np.array,
    ):
        normalized_log = -0.5 * np.log(2 * np.pi * variances)
        normalized_diff = (X - means) ** 2 / (2 * variances)
        log_likelihood = np.sum(normalized_log - normalized_diff, axis=1)
        log_likelihood = prior + log_likelihood

        return log_likelihood

    def predict(self, X):
        if self.fitted is False:
            raise ValueError("You should fit before predict.")

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            prior = self.priors[i]
            means = self.means[i]
            variances = self.variances[i]

            log_likelihood = self.log_gaussian(X, prior, means, variances)
            log_probs[:, i] = prior + log_likelihood

        return self.classes[np.argmax(log_probs, axis=1)]


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
