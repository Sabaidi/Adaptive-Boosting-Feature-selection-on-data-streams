from BG import BG
from Oracle import Oracle
from ABFS import ABFS
import time
from KNNClassifier import KNNClassifier
from SVMClassifier import SVMClassifier
from HTClassifier import HTClassifier
from NBClassifier import NBClassifier
from SelectionAccuracy import SelectionAccuracy
from SimpleClassifier import SimpleClassifier
import random
import numpy as np
from copy import deepcopy

class TrainingProcedure(object):

    def __init__(self):

        random.seed(time.time())
        self.timesteps = 20000
        self.numRelevantFeatures = 3
        self.numFeatures         = 100
        self.featureGenerator    = BG(self.numFeatures, self.numRelevantFeatures, 0.05)
        self.featureSelector     = Oracle(self.featureGenerator)
        self.featureSelector     = ABFS(Oracle(self.featureGenerator), self.numFeatures)
        #self.classifier          = KNNClassifier()
        #self.classifier          = SVMClassifier()
        #self.classifier          = HTClassifier()
        self.classifier          = NBClassifier()
        #self.featureSelector     = SimpleClassifier(self.classifier)
        self.selectionAccuracy   = SelectionAccuracy(self.featureGenerator, self.featureSelector)


        # first drift
        self.featureGenerator.initiateDrift(1000, 6666)

    def run(self):

        sa    = 0
        saSum = 0
        for i in range(self.timesteps):
            
            # second drift
            if i == 9999:
                self.featureGenerator.initiateDrift(1000, 13333)

            data, label = self.featureGenerator.next()
            self.featureSelector.train(deepcopy(data), label)

            drift = False
            if self.featureSelector.resetClassifier():
                self.classifier.reset()
                drift = True

            relevantFeatures = self.featureSelector.getRelevantFeatures()

            if len(relevantFeatures) > 0:
                self.classifier.add(np.array(data)[relevantFeatures], label)

            sa    = sa * 0.99 + self.selectionAccuracy.getAccuracy()
            saSum = saSum * 0.99 + 1

            print("time: ", i, ", accuracy: ", self.classifier.accuracy(), ", numFeatures: ", len(relevantFeatures), ", SA: ", sa / saSum, ", drift: ", int(drift))

            #if i % 100 == 0:
            #    print("time: ", i, ", accuracy: ", self.classifier.accuracy(), ", relevant features: ", relevantFeatures, ", SA: ", sa / saSum)
    
            


train = TrainingProcedure()
train.run()

