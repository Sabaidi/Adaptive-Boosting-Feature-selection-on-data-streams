class SelectionAccuracy(object):
    
    def __init__(self, featureGenerator, featureSelector, gamma = 0.5):
        
        self.featureSelector  = featureSelector
        self.featureGenerator = featureGenerator
        self.gamma            = gamma

    
    def getAccuracy(self):
        numFeatures = self.featureGenerator.getNumFeatures()

        relevantFeatures = self.featureGenerator.getRelevantFeatures()
        selectedFeatures = self.featureSelector.getRelevantFeatures()
        
        numRelevant = 0
        for f in relevantFeatures:
            if f in selectedFeatures:
                numRelevant += 1

        numIrelevant = 0

        for f in selectedFeatures:
            if f not in relevantFeatures:
                numIrelevant += 1

        return self.gamma * (numRelevant / len(relevantFeatures)) + \
               (1 - self.gamma) * (1 - numIrelevant / (numFeatures - len(relevantFeatures)))
            
