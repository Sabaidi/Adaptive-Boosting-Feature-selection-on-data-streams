from sklearn import svm
import numpy as np

class SVMClassifier(object):

    def reset(self):
        self.data    = np.zeros((500, 1))
        self.labels  = np.zeros(500)
        self.weights = np.ones(500)
        for i in range(25):
            self.labels[i] = 1
        self.learner.fit(self.data, self.labels, self.weights)


    def __init__(self):
        
        self.sumCorrect = 0
        self.numSamples = 0
        self.learner    = svm.SVC()
        self.reset()

    def add(self, features, label, weight = 1):

        while self.data.shape[1] < len(features):
            self.data = np.concatenate((self.data, np.zeros((500, 1))), axis=1)
            self.learner.fit(self.data, self.labels)

        pred = int(self.learner.predict(np.array([features], dtype=int)))
        if pred == int(label):
            self.sumCorrect = self.sumCorrect * 0.99 + 1
        else:
            self.sumCorrect = self.sumCorrect * 0.99

        self.numSamples = self.numSamples * 0.99 + 1
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1]  = np.array(features, dtype=int)

        self.labels = np.roll(self.labels, -1)
        self.labels[-1]  = np.array(label)

        self.weights = np.roll(self.weights, -1)
        self.weights[-1]  = np.array(weight)

        self.learner.fit(self.data, self.labels, self.weights)

    def predict(self, features):
        while self.data.shape[1] < len(features):
            self.data = np.concatenate((self.data, np.zeros((500, 1))), axis=1)
            self.learner.fit(self.data, self.labels)

        return int(self.learner.predict(np.array([features], dtype=int)))

    def accuracy(self):
        if self.numSamples == 0:
            return 0

        return self.sumCorrect / self.numSamples

