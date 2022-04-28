import numpy as np

class PurityMetric(object):

    def __init__(self, numFeatures):
        self.zeroTotal = 0
        self.oneTotal  = 0
    
        self.oneSame    = np.zeros(numFeatures)
        self.zeroSame   = np.zeros(numFeatures) 
        self.numSamples = 0

    def add(self, features, label):
        
        self.numSamples += 1

        if label == 1:
            self.oneTotal += 1
        else:
            self.zeroTotal += 1
    

        for i in range(len(features)):
            if label == 1 and features[i] == 1:
                self.oneSame[i] += 1

            if label == 0 and features[i] == 0:
                self.zeroSame[i] += 1

    def len(self):
        return self.numSamples

    def giniIndex(self, i):
        
        gZero = 1 - (self.zeroSame[i] / self.zeroTotal)**2 - ((self.zeroTotal - self.zeroSame[i]) / self.zeroTotal)**2
        gOne  = 1 - (self.oneSame[i]  / self.oneTotal)**2  - ((self.oneTotal  - self.oneSame[i])  / self.oneTotal)**2

        return gZero * (self.zeroTotal / (self.zeroTotal + self.oneTotal)) + \
               gOne  * (self.oneTotal  / (self.zeroTotal + self.oneTotal))




    

