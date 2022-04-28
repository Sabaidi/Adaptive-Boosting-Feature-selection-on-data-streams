from DecisionStump import DecisionStump
from BoostingUnit import BoostingUnit
from skmultiflow.drift_detection.adwin import ADWIN
import numpy as np

class ABFS(object):

    def __init__(self, oracle, numFeatures):
        
        self.dsCandidate      = DecisionStump(0.01, 500, numFeatures)
        self.numFeatures      = numFeatures
        self.boosingUnits     = []
        self.selectedFeatures = []
        self.oracle           = oracle
        self.resetLearner     = False

    def appendFeature(self, featureIndex):
        for i in range(len(self.selectedFeatures)):
            if featureIndex >= self.selectedFeatures[i]:
                featureIndex += 1

        self.selectedFeatures.append(featureIndex)

    def train(self, features, label):

        weight = 1
        iDrift = -1
        self.selectedFeatures = []
        self.resetLearner     = False

        for i in range(len(self.boosingUnits)):
            weight = self.boosingUnits[i].add(features, label, weight);

            #drift recognised
            if self.boosingUnits[i].resetClassifier():
                iDrift = i
                break

            self.appendFeature(self.boosingUnits[i].getSelectedFeature())
            del features[self.boosingUnits[i].getSelectedFeature()]

        if iDrift == -1:

            self.dsCandidate.train(features, label, weight)
            if self.dsCandidate.getSelectedFeature() != -1:
                self.boosingUnits.append(BoostingUnit(self.dsCandidate, ADWIN()))
                self.dsCandidate = DecisionStump(0.01, 500, self.numFeatures - len(self.boosingUnits))
        else:
            print("Drift detected at ", iDrift)
            while len(self.boosingUnits) > iDrift:
                del self.boosingUnits[-1]

            self.resetLearner = True
            self.dsCandidate  = DecisionStump(0.01, 500, self.numFeatures - len(self.boosingUnits))

    def select(self):
        return self.selectedFeatures

    def resetClassifier(self):
        return self.resetLearner

    def getRelevantFeatures(self):
        return self.selectedFeatures
