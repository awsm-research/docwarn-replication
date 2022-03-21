########################################################################################
# SETUP
########################################################################################

this_file = gsub("--file=", "", commandArgs()[grepl("--file", commandArgs())])
if (length(this_file) > 0){
  wd <- paste(head(strsplit(this_file, '[/|\\]')[[1]], -1), collapse = .Platform$file.sep)
} else if (Sys.getenv("RSTUDIO") == "1") {
  wd <- dirname(rstudioapi::getSourceEditorContext()$path)
} else {
  wd <- paste0(getwd(), '/Rscript')
}
setwd(wd)
source("util.R")
source("model_util.R")

########################################################################################


library(dplyr)
library(plyr)
library(tidyr)
library(ggplot2)
library(fpc)
library(Rnalytica)
library(rms)
library(randomForest)
library(e1071)
library(car)
library(ROCR)
library(caret)
library(OneR)
library(data.table)

dataMode = 'cross'
variation = ''
averageOutputDf <- data.frame()
for (project in infoChgList) {
  resultDf = data.frame()
  # get columns to be used for this project
  for (i in 0:getNumberOfRound(dataMode)) {
    r = getRandomRoundCvRound(i)
    randomRound = r[1]
    cvRound = r[2]
    print(paste0(project, "-", i))
    
    modelPath = paste0("../../data/modelResult/models", variation, "/", dataMode, "/", project)
    load(paste0(modelPath, "_rf_", randomRound, "_", cvRound, ".rda"))
    
    tempDf = getDf(dataMode, project, randomRound, cvRound, forRandomForest = TRUE)
    trainingDf = tempDf$train
    testingDf = tempDf$test
    selectedColumns = c(rownames(rf$importance))
    
    #random guessing
    randomPredicted = randomGuess(length(testingDf$y_cosine))
    randomResult = calculatePRF(randomPredicted, testingDf$y_cosine, 0.5, "random")
    names(randomResult) = paste0("random_", names(randomResult))
    #oneR
    tempDf = select(trainingDf, c(selectedColumns, 'y_cosine'))
    data.bin <- optbin(tempDf)
    model.OneR <- OneR(formula(paste("y_cosine" , " ~ " , paste(selectedColumns,collapse="+") )), data = trainingDf, ties.method = c("first", "chisq"))
    predictedOneR = as.integer(predict(model.OneR, testingDf[selectedColumns])) - 1
    onerResult = calculatePRF(predictedOneR, testingDf$y_cosine, 0.5, "oner")
    names(onerResult) = paste0("oner_", names(onerResult))
    
    #LOAD DATA
    tempDf = getDf(dataMode, project, randomRound, cvRound, forRandomForest = TRUE)
    trainingDf = tempDf$train
    testingDf = tempDf$test
    selectedColumns = c(rownames(rf$importance))
    
    #RF AUC
    trainingDf$y_predicted = data.frame(predict(rf, trainingDf[,selectedColumns], type="prob"))$X1
    cutoff = getCutOff(trainingDf)
    rfPredicted = data.frame(predict(rf, testingDf[,selectedColumns], type="prob"))$X1
    rfResult = calculatePRF(rfPredicted, testingDf$y_cosine, cutoff, "rf")
    names(rfResult) = paste0("rf_", names(rfResult))
    
    resultDf = rbind(resultDf, c(rfResult, randomResult, onerResult))
  }
  
  writeCsv(resultDf, paste0("../../data/modelResult/performance_pdc_c/", project, "_", dataMode, variation, "_all.csv"))
  
  outputList = list(project=project)
  for (col in colnames(resultDf)) {
    outputList[col] = mean(na.omit(resultDf[,col]))
  }
  averageOutputDf = rbind(averageOutputDf, data.frame(outputList))
}
writeCsv(averageOutputDf, paste0("../../data/modelResult/performance_pdc_c/", dataMode, variation, "_average.csv"))







