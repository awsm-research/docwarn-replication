import os
home = os.getcwd().replace('code/model', '')
projName = "code"
dataName = "data"

code = home + '/' + projName + "/"
data = home + '/' + dataName + "/"
modelResult = '/modelResult/'
dataRevertedCleaned = "/data_reverted_cleaned/"
features = "/features/"
issues = "/issues/"
trainingData = "/trainingData/"

cross = "/cross/"
merge = "/merge/"


def getTrainingDataPath_cross(projectKey):
    return data + trainingData + 'cross/' + projectKey
