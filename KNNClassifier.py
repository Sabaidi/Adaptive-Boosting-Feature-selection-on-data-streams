from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

class KNNClassifier(object):

    def reset(self):
        self.data   = np.zeros((500, 1))
        self.labels = np.zeros(500)
        self.learner.fit(self.data, self.labels)


    def __init__(self):
        
        self.sumCorrect = 0
        self.numSamples = 0
        self.learner    = KNeighborsRegressor(n_neighbors=2)
        self.reset()

    def add(self, features, label):
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
        self.data[-1]  = np.array(features).astype(int)

        self.labels = np.roll(self.labels, -1)
        self.labels[-1]  = np.array(label).astype(int)

        self.learner.fit(self.data, self.labels)
        #print(self.data.shape)
        #print(self.labels.reshape((50,1)).shape)
        #print(np.concatenate((self.data, self.labels.reshape((50,1))), axis=1))
        #print(np.array(features).astype(int), np.array(label).astype(int) )

    def accuracy(self):
        if self.numSamples == 0:
            return 0

        return self.sumCorrect / self.numSamples

