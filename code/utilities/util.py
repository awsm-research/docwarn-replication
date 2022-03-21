import csv
import threading
from datetime import datetime

import numpy
import pandas
import pytz
import urllib3
from nltk.corpus import stopwords

from utilities import Log

lock = threading.Lock()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
stops = set(stopwords.words("english"))

file_type = ".csv"
VALUE_NOT_EXIST = numpy.nan

# cosineList = [0.80, 0.85, 0.90, 0.95, 1]
cosineList = [0.80, 0.90, 1]


def write_csv_dataFrame(dataFrame, fileName):
    createDirIfNotExist(fileName)
    lock.acquire()
    Log.i("Write " + fileName)
    dataFrame.to_csv(fileName + ".csv", encoding='utf-8', index=False, sep=",", quotechar='"',
                     quoting=csv.QUOTE_NONNUMERIC)
    lock.release()


def createDirIfNotExist(filePath):
    pathSplit = filePath.split('/')[:-1]
    folderPath = '/'.join([x for x in pathSplit])
    try:
        import os
        os.makedirs(folderPath)
    except FileExistsError:
        # directory already exists
        pass


def read_csv_pandas(filePath):
    import pandas as pd
    # Log.i("read " + filePath)
    if not filePath.endswith('.csv'):
        filePath = filePath + ".csv"
    df = pd.read_csv(filePath, sep=',', quotechar='"', quoting=True)
    return df


def fillNanWithEmptyString(df: pandas.DataFrame):
    return df.replace(numpy.nan, '', regex=True)


def fillNanWithZeroDf(df: pandas.DataFrame):
    return df.replace(numpy.nan, 0, regex=True)


def dateTimeToUtc(dt: datetime):
    return dt.astimezone(pytz.utc)


def nrow(df):
    return len(df.index)


def dateFromRaw(time: str):
    # 2014-05-26T07:32:36.000+0000
    temp = time.split('.')
    timezone = temp[1][-5:]  # +0000
    timezone = timezone[:3] + ':' + timezone[3:]  # 2014-05-26T07:32:36+00:00
    import datetime
    dt = datetime.datetime.fromisoformat(temp[0] + timezone)
    return dateTimeToUtc(dt)



def dateToMillis(dt: datetime):
    return dt.timestamp() * 1000


def nth_repl_all(s, sub, repl, nth):
    find = s.find(sub)
    # loop util we find no match
    i = 1
    while find != -1:
        # if i  is equal to nth we found nth matches so replace
        if i == nth:
            s = s[:find] + repl + s[find + len(sub):]
            i = 0
        # find + len(sub) + 1 means we start after the last match
        find = s.find(sub, find + len(sub) + 1)
        i += 1
    return s


urlRegexCoverAll = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
urlRegexStrongOnly = r"(https?://\S+)"


def decontracted(phrase):
    # https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python
    # specific
    import re
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    for original, replacement in load_dict_contractions().items():
        phrase = phrase.replace(' ' + original + ' ', ' ' + replacement + ' ')
    phrase = phrase.replace('e.g.', 'for example')
    phrase = phrase.replace('i.e.', 'which is')
    return phrase


def load_dict_contractions():
    return {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "'cause": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        " em ": " them ",
        "everyone's": "everyone is",
        "gimme": "give me",
        "gonna": "going to",
        "gotta": "got to",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "he've": "he have",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "I've": "I have",
        "kinda": "kind of",
        "let's": "let us",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "might've": "might have",
        "mustn't": "must not",
        "must've": "must have",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shalln't": "shall not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why's": "why is",
        "won't": "will not",
        "wouldn't": "would not",
        "would've": "would have",
        "y'all": "you all",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
    }


def appendToFile(newDataDf, filePath):
    # filePath = paths.data + paths.deepAucReport + "report"
    createDirIfNotExist(filePath + ".csv")
    try:
        savedData = read_csv_pandas(filePath)
    except:
        savedData = pandas.DataFrame()

    # newData = pandas.DataFrame.from_records([reportParams])
    savedData = savedData.append(newDataDf)
    write_csv_dataFrame(savedData, filePath)


def getPredResult(y, y_predicted, threshold=0.5):
    from sklearn.metrics import roc_auc_score, f1_score, recall_score, \
        precision_score, accuracy_score, balanced_accuracy_score
    # https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
    try:
        auc = roc_auc_score(numpy.array(y, dtype=int), y_predicted)
    except:
        auc = -1
    f1 = f1_score(y, y_predicted > threshold, pos_label=True)
    precision = precision_score(y, y_predicted > threshold, pos_label=True)
    recall = recall_score(y, y_predicted > threshold, pos_label=True)
    accuracy = accuracy_score(y, y_predicted > threshold)
    balancedAccuracy = balanced_accuracy_score(y, y_predicted > threshold)

    return auc, f1, precision, recall, accuracy, balancedAccuracy


def getCvFileName(projectKey, cosine, randomRound, cvRound):
    return projectKey + "_" + str(int(cosine * 100)) + "_" + str(randomRound) + "_" + str(cvRound)
