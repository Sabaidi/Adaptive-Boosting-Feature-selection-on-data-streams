from sklearn import svm
import numpy as np
from PurityMetric import PurityMetric
from HTClassifier import HTClassifier
from SVMClassifier import SVMClassifier
from NBClassifier import NBClassifier

class DecisionStump(object):

    def __init__(self, selectionThreshold, gracePeriod, numFeatures):
        self.gracePeriod        = gracePeriod
        self.selectionThreshold = selectionThreshold
        self.selectedFeature    = -1
        #self.learner            = HTClassifier()
        #self.learner            = SVMClassifier()
        self.learner            = NBClassifier()
        self.metric             = PurityMetric(numFeatures)

    def predict(self, features):
        return self.learner.predict(features)

    def train(self, features, label, weight):
        self.metric.add(features, label)
        self.learner.add(features, label, weight)

        if self.metric.len() > self.gracePeriod:
            bestFeatureValue   = 1
            bestFeature        = -1
            secondBestFeature  = -1
            secondBestValue    = 1
            worstValue         = 0

            for i in range(len(features)):
                value = self.metric.giniIndex(i)
                if value < bestFeatureValue:
                    secondBestFeature  = bestFeature
                    secondBestValue    = bestFeatureValue
                    bestFeature        = i
                    bestFeatureValue   = value
                if value > worstValue:
                    worstValue = value

            hoeffdingBound = np.sqrt(np.log(1/1e-6) / (2 * self.metric.len()))
            #print("hoeffdingBound: ", hoeffdingBound, " / ", abs(bestFeatureValue - secondBestValue), ", bestFeatureValue: ", bestFeatureValue, ", secondBestValue: ", secondBestValue, ", selectionThreshold: ", self.selectionThreshold)
            if (abs(bestFeatureValue - secondBestValue) > hoeffdingBound or \
               abs(secondBestValue - worstValue) > 0.05) and \
               bestFeatureValue > self.selectionThreshold:

               #hoeffdingBound < 0.05) and \
#                print("hoeffdingBound: ", hoeffdingBound, " / ", abs(bestFeatureValue - secondBestValue), ", bestFeatureValue: ", bestFeatureValue, ", secondBestValue: ", secondBestValue, ", selectionThreshold: ", self.selectionThreshold)
                self.selectedFeature = bestFeature

    def getSelectedFeature(self):
        return self.selectedFeature

