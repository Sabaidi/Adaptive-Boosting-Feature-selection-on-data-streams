import random
import utils
import time

class BG(object):
    
    def __init__(self, numFeatures, numRelevantFeatures, noise):
        random.seed(time.time() * 19)
        self.timestep = 0
        self.numFeatures = numFeatures
        self.noise = noise
        self.relevantFeatures = []
        self.driftFeatures = []
        self.driftWindow = 0
        self.driftTime = 0
        self.drifted = False

        
        # select random relevant features
        for i in range(numRelevantFeatures):
            index = random.randrange(0, numFeatures)
            while index in self.relevantFeatures:
                index = random.randrange(0, numFeatures)

            self.relevantFeatures.append(index)

    def getNumFeatures(self):
        return self.numFeatures

    def initiateDrift(self, driftWindow, driftTime):
        
        self.driftWindow = driftWindow
        self.driftTime   = driftTime

        if len(self.driftFeatures) > 0:
            self.relevantFeatures = self.driftFeatures;

        self.driftFeatures = []
        
        for i in range(len(self.relevantFeatures)):
            index = random.randrange(0, self.numFeatures)
            while index in self.driftFeatures: # or index in self.relevantFeatures:
                index = random.randrange(0, self.numFeatures)

            self.driftFeatures.append(index)
    
    # returns the drift propagility 
    def driftPropability(self):
        if len(self.driftFeatures) == 0:
            return 0

        #return 1.0/(1 + utils.exp(-1.0 / self.driftWindow * (self.timestep - self.driftTime)))
        if self.timestep < self.driftTime - self.driftWindow / 2:
            return 0

        if self.timestep > self.driftTime + self.driftWindow / 2:
            return 1

        return (self.timestep - (self.driftTime - self.driftWindow))/self.driftWindow

    
    def getRelevantFeatures(self):
        if self.drifted:
            return self.driftFeatures;
        return self.relevantFeatures;
    

    def next(self):

        features = []
        label      = bool(random.getrandbits(1))
        driftLabel = bool(random.getrandbits(1))

        self.drifted = False
        if random.random() < self.driftPropability():
            self.drifted = True

        for i in range(self.numFeatures):
            features.append(bool(random.getrandbits(1)))

        if self.drifted:
            label = driftLabel

        if label == True:
            for index in self.relevantFeatures:
                features[index] = True
        else:
            index = self.relevantFeatures[random.randrange(0, len(self.relevantFeatures))]
            features[index] = False

        if random.random() < self.noise:
            label = not label
        
        # end the feature drift
        if self.driftPropability() > 0.99999:
            self.relevantFeatures = self.driftFeatures
            self.driftFeatures = []
            self.drifted = False

        self.timestep += 1
        return (features, label)
