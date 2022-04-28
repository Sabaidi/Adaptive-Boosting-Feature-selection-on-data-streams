from DecisionStump import DecisionStump
from BoostingUnit import BoostingUnit
from skmultiflow.drift_detection.adwin import ADWIN
import numpy as np

class SimpleClassifier(object):

    def __init__(self, classifier):
        
        self.classifier    = classifier
        self.driftDetector = ADWIN()
        self.selectedFeatures = []

    def train(self, features, label):

        self.resetLearner     = False
        self.selectedFeatures = range(0, len(features))

        if self.classifier.predict(features) == int(label):
            self.driftDetector.add_element(0)
        else:
            self.driftDetector.add_element(1)

    def select(self):
        return self.selectedFeatures

    def resetClassifier(self):
        return self.driftDetector.detected_change()

    def getRelevantFeatures(self):
        return self.selectedFeatures
