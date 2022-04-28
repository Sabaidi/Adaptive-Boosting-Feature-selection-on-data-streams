import numpy as np
from skmultiflow.bayes import NaiveBayes

class NBClassifier(object):

    def reset(self):
        self.learner = NaiveBayes()

    def __init__(self):
        
        self.sumCorrect = 0
        self.numSamples = 0
        self.numFeatures = 0
        self.reset()

    def add(self, features, label, weight = 1):

        if self.numFeatures < len(features):
            self.reset();
            self.numFeatures = len(features)

        if self.predict(features) == int(label):
            self.sumCorrect = self.sumCorrect * 0.99 + 1
        else:
            self.sumCorrect = self.sumCorrect * 0.99

        self.numSamples = self.numSamples * 0.99 + 1
        self.learner.partial_fit(np.array([features], dtype=int), np.array([label], dtype=int), sample_weight=np.array([weight]))

    def predict(self, features):
        return int(self.learner.predict(np.array([features], dtype=int)))

    def accuracy(self):
        if self.numSamples == 0:
            return 0

        return self.sumCorrect / self.numSamples

