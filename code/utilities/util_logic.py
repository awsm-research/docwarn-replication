import os
import sys


sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(__file__))

import numpy
import urllib3
from bs4 import BeautifulSoup
from utilities.util import appendToFile

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas
import paths
from utilities import util, field, Log


# DONT CHANGE, IF NEED CHANGE, MAKE SURE CHANGE REGEX TOO
TAG_CODE = "[code]"
TAG_LINK = "[link]"
TAG_VERSION = "[version]"


def cleanJiraString(cleaningStr, forBert=False, doLower=True):
    cleaningStr = util.decontracted(cleaningStr)
    if cleaningStr.startswith('.'):
        cleaningStr = cleaningStr[1:]
    cleaningStr = cleaningStr.strip()

    cleaningStr = BeautifulSoup(cleaningStr, features="lxml").get_text()  # extract html
    cleaningStr = cleaningStr.replace(u'\xa0', u' ') \
        .replace('h1.', '').replace('h2.', '').replace('h3.', '') \
        .replace('h4.', '').replace('h5.', '').replace('h6.', '') \
        .replace('bq.', '').replace('{{', '').replace('}}', '') \
        .replace('{html}', '').replace('{panel}', '') \
        .replace('(', '').replace(')', '') \
        .replace('. . .', '').replace('...', '')

    # code
    import re
    cleaningStr = re.sub(r"({code.*?}+)", "{code}", cleaningStr)
    cleaningStr = util.nth_repl_all(cleaningStr, "{code}", "{/code}", 2)  # replace every second {code} with {/code}
    cleaningStr = re.sub(r"{code}(.*?){/code}", " " + TAG_CODE + " ", cleaningStr)  # replace each block of {code} ... {/code} with TAG_CODE

    # remove JIRA link format [TEXT|LINK]
    urls = re.findall('\[.*?\|https?.*?\]', cleaningStr)  # case [TEXT|LINK]
    for url in urls:
        tempUrl = url.replace('[', '')
        tempUrl = tempUrl.replace(']', '')
        urlSplit = tempUrl.split('|')
        cleaningStr = cleaningStr.replace(url, urlSplit[0] + ' ' + urlSplit[1])

    # replace link http://www.google.com with "googlecom"
    from tldextract import tldextract
    urls = re.findall(util.urlRegexStrongOnly, cleaningStr)
    for url in urls:
        extractedUrl = tldextract.extract(url)
        cleaningStr = cleaningStr.replace(url, extractedUrl.domain + extractedUrl.suffix)


    cleaningStr = re.sub(r"[\{\}]+", "", cleaningStr)

    lines = cleaningStr.splitlines()
    lineCleanedStr = ''
    for l in lines:
        if len(l) > 0:
            l = l.strip()
            if not l.endswith('.'):
                l = l + "."
        lineCleanedStr = lineCleanedStr + " " + l
    cleaningStr = lineCleanedStr.strip()
    cleaningStr = re.sub(r"[\r\n\t]+", " ", cleaningStr)

    split = cleaningStr.split(' ')
    for i in range(len(split)):
        if split[i] != TAG_CODE and split[i] != TAG_LINK and split[i] != TAG_VERSION:
            if not forBert:
                split[i] = re.sub(r"[^a-zA-Z0-9]+", ' ', split[i])
            if forBert:
                split[i] = re.sub(r"[^a-zA-Z0-9.]+", ' ', split[i])
    cleaningStr = " ".join(split)

    # cleaningStr = re.sub(r"[^a-zA-Z0-9\._\s\-]+", '', cleaningStr)
    cleaningStr = re.sub(r"[\s]+", ' ', cleaningStr)
    cleaningStr = cleaningStr.replace(' . ', '. ')
    cleaningStr = cleaningStr.replace('..', '.')

    if forBert:
        for i in range(5):
            cleaningStr = cleaningStr.strip()
            if cleaningStr.endswith(' .'):
                cleaningStr = cleaningStr[:-2] + '.'
            if cleaningStr.endswith('..'):
                cleaningStr = cleaningStr[:-2] + '.'
        if not cleaningStr.endswith('.'):
            cleaningStr = cleaningStr + '.'

    if doLower:
        return cleaningStr.lower()
    else:
        return cleaningStr




def calculateAndSavePerformance(results, project, fileName):
    aucList = []
    f1List = []
    precList = []
    recallList = []
    accList = []
    baccList = []

    outputListDf = pandas.DataFrame()
    for resultCount in range(len(results)):
        result = results[resultCount]
        # r, f, auc, f1, precision, recall, acc, bacc
        r = result[0]
        f = result[1]
        auc = result[2]
        f1 = result[3]
        precision = result[4]
        recall = result[5]
        acc = result[6]
        bacc = result[7]

        if f1 == -1:
            # in case training has no positive cases, training cant be done. We skip
            continue

        tempOutput = pandas.DataFrame.from_records(
            [{field.PROJECT: project.key, field.RANDOM_ROUND: r, field.FOLD: f,
              'auc': auc, 'f1': f1, 'precision': precision,
              'recall': recall, 'acc': acc, 'bacc': bacc}])
        outputListDf = outputListDf.append(tempOutput)

        if auc != -1:
            aucList.append(auc)
        f1List.append(f1)
        precList.append(precision)
        recallList.append(recall)
        accList.append(acc)
        baccList.append(bacc)

    successRounds = len(outputListDf)
    averageOutput = None
    if successRounds > 0:
        averageAuc = sum(aucList) / len(aucList)
        averageF1 = sum(f1List) / len(f1List)
        averagePrecision = sum(precList) / len(precList)
        averageRecall = sum(recallList) / len(recallList)
        averageAcc = sum(accList) / len(accList)
        averageBacc = sum(baccList) / len(baccList)

        # save average performance
        averageOutput = pandas.DataFrame.from_records([{field.PROJECT: project.key,
                                                                  'successRounds': str(successRounds),
                                                                  'auc': averageAuc, 'f1': averageF1,
                                                                  'precision': averagePrecision,
                                                                  'recall': averageRecall,
                                                                  'acc': averageAcc, 'bacc': averageBacc}])

        Log.i(project.key + "\n" + str(averageOutput))
        appendToFile(outputListDf, paths.data + paths.modelResult + fileName + "_all")
    return averageOutput



def getDynamicThreshold(trainDf, labelKey):
    trainPosDf = trainDf[trainDf[labelKey] == True]
    trainPosQ1 = numpy.percentile(trainPosDf['predicted'], 25)
    trainNegDf = trainDf[trainDf[labelKey] == False]
    trainNegQ3 = numpy.percentile(trainNegDf['predicted'], 75)
    threshold = (trainPosQ1 + trainNegQ3) / 2
    return threshold



def prepareBertText(revertedDf, trainDf, testDf, doLower=True):
    revertedDf = util.fillNanWithEmptyString(revertedDf)
    revertedDf[field.TEXT] = revertedDf[field.SUMM] + ".\n" + revertedDf[field.DESC]
    revertedDf[field.TEXT] = revertedDf[field.TEXT].apply(lambda x: cleanJiraString(x, True, doLower))
    trainDf = trainDf[[field.ISSUE_KEY, field.LABEL, field.RANDOM_ROUND, field.CV_ROUND]].merge(revertedDf[[field.ISSUE_KEY, field.TEXT]], on=field.ISSUE_KEY, how='left')
    testDf = testDf[[field.ISSUE_KEY, field.LABEL, field.RANDOM_ROUND, field.CV_ROUND]].merge(revertedDf[[field.ISSUE_KEY, field.TEXT]], on=field.ISSUE_KEY, how='left')
    return trainDf, testDf


def runAutoSpearman(featureDf, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=2)  # util.getNumberOfCores())
    model.fit(featureDf, y)
    from pyexplainer import pyexplainer_pyexplainer
    pyExplainer = pyexplainer_pyexplainer.PyExplainer(featureDf, y, featureDf.columns, field.LABEL,
                                                      model, [False, True])
    pyExplainer.auto_spearman()
    selectedColumns = pyExplainer.X_train.columns
    return selectedColumns


def normalizeNumericalColumn(df):
    from sklearn import preprocessing
    for col in field.newFeatureList():
        minMaxScaler = preprocessing.MinMaxScaler()
        x_scaled = minMaxScaler.fit_transform(df[col].values.reshape(-1, 1))
        df[col] = x_scaled
    return df


def getEmbeddingForNumericalData(df, selectedColumns, allColumns):
    numDf = df[[col for col in selectedColumns if col in allColumns]]
    from tensorflow.keras import Input
    inp_num_data = Input(shape=(numDf.shape[1],), name='numerical_input')
    return inp_num_data


def convertDfToTensor(df):
    import tensorflow as tf
    return tf.constant(df, dtype=tf.float32, shape=df.shape)


def assertNoIntersect(s1: pandas.Series, s2: pandas.Series):
    assert len(set(s1.to_list()).intersection(set(s2.to_list()))) == 0

