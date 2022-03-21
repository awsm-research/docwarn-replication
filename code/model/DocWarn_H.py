import os
import sys


sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(__file__))

from tensorflow import keras
import projects
import pandas
import paths
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utilities import util, field, Log, util_logic
from transformers import RobertaTokenizerFast, AutoConfig, TFRobertaModel
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import numpy
from keras import backend as K
from copy import deepcopy

MAX_LENGTH = 256
tokenizerPath = 'distilroberta-base'
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizerPath)
Log.i('Tokenizer = ' + tokenizerPath)
modelPath = '../distrilroberta-base-jira'

# Configure BERT initialization
config = AutoConfig.from_pretrained(modelPath)
config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2
config.output_hidden_states = True
transformerOrigin = TFRobertaModel.from_pretrained(modelPath, config=config, from_pt=True)
for layer in transformerOrigin.layers:
    layer.trainable = False
Log.i('Tokenizer = ' + modelPath)

def batch_encode(tokenizer, texts, batch_size=256, max_length=MAX_LENGTH):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed 
    into a pre-trained layer_norm_epstransformer model.
    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""
    input_ids = []
    attention_mask = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length',  # implements dynamic padding
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])
    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


def runForProjectList():
    averageOutputDf = pandas.DataFrame()
    fileName = 'pdc_h_cross'

    for project in projects.infoChgList:
        dataDf = util.read_csv_pandas(paths.getTrainingDataPath_cross(project.key))
        randomRounds = dataDf[field.RANDOM_ROUND].unique()
        cvRounds = dataDf[field.CV_ROUND].unique()
        dataDf = dataDf.rename(columns={field.Y_COSINE: field.LABEL})
        dataDf[field.LABEL] = dataDf[field.LABEL].astype(int)

        revertedDf = util.read_csv_pandas(paths.data + paths.dataRevertedCleaned + project.key)
        revertedDf = util.fillNanWithEmptyString(revertedDf)

        dataDf = dataDf.merge(revertedDf[[field.ISSUE_KEY, field.TEXT]], how="left", on=field.ISSUE_KEY)
        allTrainDf = dataDf[dataDf['type'] == 'train']
        allTestDf = dataDf[dataDf['type'] == 'test']

        # randomRound = 0
        # cvRound = 4
        results = []
        for randomRound in randomRounds:
            for cvRound in cvRounds:
                trainDf = allTrainDf[(allTrainDf[field.RANDOM_ROUND] == randomRound) & (allTrainDf[field.CV_ROUND] == cvRound)]
                testDf = allTestDf[(allTestDf[field.RANDOM_ROUND] == randomRound) & (allTestDf[field.CV_ROUND] == cvRound)]
                util_logic.assertNoIntersect(trainDf[field.ISSUE_KEY], testDf[field.ISSUE_KEY])
                roundFileName = fileName + "_" + str(randomRound) + str(cvRound)
                Log.i("RUN " + project.key + " " + str(roundFileName))
                result = runBert(trainDf, testDf, project, roundFileName, randomRound, cvRound)
                results.append(result)
        projectAverageOutputRow = util_logic.calculateAndSavePerformance(results, project, fileName)
        if projectAverageOutputRow is not None:
            averageOutputDf = averageOutputDf.append(projectAverageOutputRow)
    util.write_csv_dataFrame(averageOutputDf, paths.data + paths.modelResult + fileName + "_average")


def runBert(trainDf, testDf, project, roundFileName, randomRound, cvRound):
    seed = (randomRound * 10) + cvRound
    if util.nrow(trainDf[trainDf[field.LABEL] == True]) <= 10 or util.nrow(testDf[testDf[field.LABEL] == True]) == 0:
        # Too few positive cases in Train or Test for this fold, ignore this window
        return randomRound, cvRound, -1, -1, -1, -1, -1, -1
    revertedDf = util.read_csv_pandas(paths.data + paths.dataRevertedCleaned + project.key)
    revertedDf = util.fillNanWithEmptyString(revertedDf)
    trainDf, testDf = util_logic.prepareBertText(revertedDf, trainDf, testDf, doLower=False)
    trainDf, validDf = train_test_split(trainDf, test_size=0.2, stratify=trainDf[field.LABEL])
    featureDf = util.read_csv_pandas(paths.data + paths.features + paths.merge + paths.cross + util.getCvFileName(project.key, 0.9, randomRound, cvRound))
    featureDf = util.fillNanWithZeroDf(featureDf)
    issueTypes = numpy.array(featureDf[field.ISSUE_TYPE].unique()).tolist()
    for issueType in issueTypes:
        featureDf.loc[featureDf[field.ISSUE_TYPE] == issueType, field.ISSUE_TYPE] = issueTypes.index(issueType)
    featureDf[field.ISSUE_TYPE] = pandas.to_numeric(featureDf[field.ISSUE_TYPE])
    priorities = numpy.array(featureDf[field.PRIORITY].unique()).tolist()
    for priority in priorities:
        featureDf.loc[featureDf[field.PRIORITY] == priority, field.PRIORITY] = priorities.index(priority)
    featureDf[field.PRIORITY] = pandas.to_numeric(featureDf[field.PRIORITY])
    featureDf = util_logic.normalizeNumericalColumn(featureDf)

    allTrainDf = trainDf.merge(featureDf, on=field.ISSUE_KEY, how='left')
    allValidDf = validDf.merge(featureDf, on=field.ISSUE_KEY, how='left')
    allTestDf = testDf.merge(featureDf, on=field.ISSUE_KEY, how='left')

    myFeatureList = field.newFeatureList()
    myFeatureList.extend([c for c in featureDf.columns if (c.startswith('components_') and not c.startswith('components_raw_'))])
    trainFeatureDf = allTrainDf[myFeatureList]

    selectedColumns = util_logic.runAutoSpearman(trainFeatureDf, allTrainDf[field.LABEL], 1)
    trainFeatureDf_vector = util_logic.convertDfToTensor(trainFeatureDf[selectedColumns])

    validFeatureDf = allValidDf[myFeatureList]
    validFeatureDf = validFeatureDf[selectedColumns]
    validFeatureDf_vector = util_logic.convertDfToTensor(validFeatureDf[selectedColumns])

    testFeatureDf = allTestDf[myFeatureList]
    testFeatureDf = testFeatureDf[selectedColumns]
    testFeatureDf_vector = util_logic.convertDfToTensor(testFeatureDf)


    X_train_ids, X_train_attention = batch_encode(tokenizer, trainDf[field.TEXT].tolist())
    X_valid_ids, X_valid_attention = batch_encode(tokenizer, validDf[field.TEXT].tolist())
    X_test_ids, X_test_attention = batch_encode(tokenizer, testDf[field.TEXT].tolist())


    transformer = deepcopy(transformerOrigin)
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=seed)
    input_ids_layer = tf.keras.layers.Input(shape=(MAX_LENGTH,), name='input_ids', dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(MAX_LENGTH,), name='input_attention', dtype='int32')
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]
    cls_token = last_hidden_state[:, 0, :]
    numericalInputLayer = util_logic.getEmbeddingForNumericalData(trainFeatureDf, selectedColumns, myFeatureList)
    conc = keras.layers.Concatenate()([numericalInputLayer, cls_token])  # tf.concat
    output = tf.keras.layers.Dense(1,
                                   activation='sigmoid',
                                   kernel_initializer=weight_initializer,
                                   kernel_constraint=None,
                                   bias_initializer='zeros'
                                   )(conc)

    LEARNING_RATE = 0.005
    EPOCHS = 20
    BATCH_SIZE = 64
    NUM_STEPS = len(trainDf.index) // BATCH_SIZE
    model = tf.keras.Model(inputs=[input_ids_layer, input_attention_layer, numericalInputLayer], outputs=output)
    model.compile(tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, restore_best_weights=True)
    # Train the model
    Log.i(X_train_ids.shape)
    Log.i(X_train_attention.shape)
    Log.i(trainFeatureDf_vector.shape)
    train_history1 = model.fit(
        x=[X_train_ids, X_train_attention, trainFeatureDf_vector],
        y=trainDf[field.LABEL].to_numpy(),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=NUM_STEPS,
        validation_data=([X_valid_ids, X_valid_attention, validFeatureDf_vector], validDf[field.LABEL].to_numpy()),
        verbose=1,
        callbacks=[overfitCallback]
    )

    trainDf['predicted'] = model.predict([X_train_ids, X_train_attention, trainFeatureDf_vector])
    threshold = util_logic.getDynamicThreshold(trainDf, field.LABEL)
    y_test_predicted = model.predict([X_test_ids, X_test_attention, testFeatureDf_vector])
    auc, f1, precision, recall, acc, bacc = util.getPredResult(testDf[field.LABEL], y_test_predicted, threshold)
    Log.i("finished " + project.key + " " + roundFileName + " auc:" + str(auc) + " f1:" + str(f1))

    K.clear_session()
    return randomRound, cvRound, auc, f1, precision, recall, acc, bacc



if __name__ == "__main__":

    tf.test.is_gpu_available()
    tf.test.is_built_with_cuda()
    runForProjectList()

