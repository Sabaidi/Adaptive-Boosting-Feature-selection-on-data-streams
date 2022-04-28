class BoostingUnit(object):
    
    def __init__(self, decisionStump, driftDetector):
        self.decisionStump = decisionStump
        self.lambdaC       = 0
        self.lambdaE       = 0
        self.driftDetector = driftDetector

    # updates this and returns the new lambda
    def add(self, features, label, weight):

        if self.decisionStump.predict(features) == int(label):
            self.lambdaC += weight
            weight *= (self.lambdaC + self.lambdaE) / (2.0 * self.lambdaC)
            self.driftDetector.add_element(0)
        else:
            self.lambdaE += weight
            weight *= (self.lambdaC + self.lambdaE) / (2.0 * self.lambdaE)
            self.driftDetector.add_element(1)
        
        return weight

    def predict(self, features):
        return self.decisionStump.predict(features)

    def getSelectedFeature(self):
        return self.decisionStump.getSelectedFeature()

    def resetClassifier(self):
        return self.driftDetector.detected_change()
