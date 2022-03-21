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
library(ScottKnottESD)
library(tidyverse)
library(mmpf)
dataMode = 'cross'
resultDf = data.frame()
for (project in infoChgList) {
  allContributionDf = data.frame()
  for (i in 0:getNumberOfRound(dataMode)) {  
    r = getRandomRoundCvRound(i)
    randomRound = r[1]
    cvRound = r[2]
    load(paste0("../../data/modelResult/models/", dataMode, "/", project, "_rf_", randomRound, "_", cvRound, ".rda"))
    
    tempDf = getDf(dataMode, project, randomRound, cvRound, forRandomForest = TRUE)
    trainingDf = tempDf$train
    testingDf = tempDf$test  
    
    componentColumns = tempDf$componentColumns
    varnames = rownames(data.frame(importance(rf)))
    trainingDf = trainingDf[c(varnames, 'y_cosine')]
    testingDf = testingDf[c(varnames, 'y_cosine')]
    
    componentColumns = componentColumns[componentColumns %in% varnames]
    varnames = varnames[!varnames %in% componentColumns]
    
    impResult = data.frame()
    for (varname in varnames) {
      value = permutationImportance(testingDf, varname, 'y_cosine', rf)
      result = data.frame(varname=varname, value=value)
      impResult = rbind(impResult, result)
    }
    value = permutationImportance(testingDf, componentColumns, 'y_cosine', rf)
    result = data.frame(varname='components', value=value)
    impResult = rbind(impResult, result)
    impResult = impResult[impResult$value > 0,]
    impResult$value = impResult$value / sum(impResult$value) * 100
    print(impResult)
    allContributionDf = rbind(allContributionDf, impResult)
  }
  expectedRounds = getNumberOfRound(dataMode) + 1
  skEsdTable = data.frame(matrix(0, ncol = 0, nrow = expectedRounds))
  for (var in unique(allContributionDf$varname)) {
    toFill = allContributionDf[allContributionDf$varname == var,]$value
    if (length(toFill) < expectedRounds) {
      amountNeeded = expectedRounds-length(toFill)
      toFill = c(toFill, rep(0:0, each=amountNeeded))
      print(paste0(var, " is missing (", amountNeeded, 'times)'))
    }
    skEsdTable[var] = toFill
  }
  sk = sk_esd(skEsdTable)
  tempDf <- tibble::rownames_to_column(data.frame(sk$groups), "varname")
  tempDf$project = project
  resultDf = rbind(resultDf, tempDf)
}
colnames(resultDf) = c("varname", "rank", "project")
writeCsv(resultDf, paste0("../../data/modelResult/varrank_", dataMode, ".csv"))

