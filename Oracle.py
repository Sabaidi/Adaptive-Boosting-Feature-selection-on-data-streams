class Oracle(object):

    def __init__(self, featureGenerator):
        self.featureGenerator = featureGenerator

    def getRelevantFeatures(self):
        return self.featureGenerator.getRelevantFeatures()

    # reset if we are inside a drift
    def resetClassifier(self):
        return self.featureGenerator.driftPropability() > 0.05 and \
               self.featureGenerator.driftPropability() < 0.95

    def add(self, var):
        return var

    def train(self, features, label, weight = 1):
        return 0
