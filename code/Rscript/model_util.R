
library(dplyr)
library(tidyr)
library(fpc)
library(Rnalytica)
library(rms)
library(randomForest)
library(e1071)
library(car)
library(ROCR)
library(caret)
library(OneR)
library(pROC)
library(DMwR)


calculatePRF <- function(predicted, result, cutoff, name) {
  predictedLabels = factor(predicted >= cutoff, levels=c(FALSE,TRUE))
  result = factor(result == 1, levels=c(FALSE,TRUE))
  if (sum(predictedLabels==FALSE) == length(predicted)) {
    acc = 0
  } else {
    xtab = table("a" = factor(predicted >= cutoff, levels=c(FALSE,TRUE)), "b" = result)
    matrix = caret::confusionMatrix(xtab, positive = "TRUE")
    acc = matrix$byClass['Balanced Accuracy']
  }
  
  if (name != "oner" && name != "random") {
    auc = auc(result, predicted, quiet = TRUE)
  } else {
    auc = 0
  }
  
  library(caret)
  p <- caret::posPredValue(predictedLabels, result, positive=TRUE)
  r <- caret::sensitivity(predictedLabels, result, positive=TRUE)
  f1 <- 2 * (p * r) / (p + r)
  
  if (is.na(p)) {
    p = 0
  }
  if (is.na(r)) {
    r = 0
  }
  if (is.na(f1)) {
    f1 = 0
  }
  
  return(list(auc = auc, 
              acc = acc,
              precision = p, 
              recall = r, 
              f = f1))
}


randomGuess = function(size) {
  return(sample(0:1, size, replace=TRUE))
}


removeVars_new <- function(df, columns) {
  ind_vars = AutoSpearman(dataset = df, metrics = columns)
  ind_vars_copy = ind_vars
  for (var in ind_vars_copy) {
    count = nrow(unique(df[var]))
    if (count <= 1) {
      ind_vars = ind_vars[in_vars != var]
    }
  }
  red <- redun(~., data=df[,ind_vars], nk=0) 
  reject_vars <- red$Out
  ind_vars <- ind_vars[!(ind_vars %in% reject_vars)]
  budgetted_DF = floor(min(nrow(df[df$y == TRUE,]), nrow(df[df$y == FALSE,]) )/15)
  return(ind_vars)
}


factorFeatures = c('issuetype', 'priority', 'has_CODE', 'has_TESTCASE', 'has_ATTACHMENT', 'has_AC', 'has_epic', 'has_subtasks', 'has_STACK')
                   #'has_OB', 'has_EB', 'has_STEP'
getDf <- function(dataMode, project, randomRound, cvRound, forRandomForest, model_formula=NA) {
  featuresDf = read.csv(paste0("../../data/features/merge/", dataMode, "/", project, "_90_", randomRound, "_", cvRound, ".csv"), stringsAsFactors = TRUE)
  cvDf = read.csv(paste0("../../data/trainingData/", dataMode, "/", project, ".csv"))
  cvDf = cvDf[cvDf$cvRound == cvRound & cvDf$randomRound == randomRound,]
  df = featuresDf[featuresDf$issue_key %in% cvDf$issue_key,]
  df[is.na(df)] <- 0
  df[df == -999] <- 0
  df = df[!grepl("^components_raw", colnames(df))]
  df = df[!grepl("^text_", colnames(df))]
  columns = colnames(df)
  
  componentColumns = columns[grepl("^components_", columns)]
  
  df[factorFeatures] <- lapply(df[factorFeatures], factor)
  df[componentColumns] <- lapply(df[componentColumns], factor)
  
  df$y_cosine <- factor(df$y_cosine)
  
  trainDf = df[df$type == 'train',]
  trainDf = trainDf[ , !names(df) %in% c('type')]
  testDf = df[df$type == 'test',]
  testDf = testDf[ , !names(df) %in% c('type')]
  if (forRandomForest) {
    for (f in factorFeatures) {
      levels(trainDf[[f]]) <- levels(df[[f]])
      levels(testDf[[f]]) <- levels(df[[f]])
      assertthat::are_equal(get_levels(trainDf, f), get_levels(testDf, f))
      
      trainUnique = unique(trainDf[[f]])
      testUnique = unique(testDf[[f]])
      if (!identical(trainUnique, testUnique)) {
        mostFrequentValueInTrain = 
          tail(names(sort(table(trainDf[[f]]))), 1)
        
        if (length(testUnique[!testUnique %in% trainUnique]) > 0) {
          for (testValueToBeReplaced in testUnique[!testUnique %in% trainUnique]) {
            testDf[[f]][testDf[[f]]==testValueToBeReplaced]<-mostFrequentValueInTrain
          }
        }
      }
    }
  } else {
    for (f in factorFeatures) {
      trainDf[f] = as.factor(trainDf[[f]])
      testDf[f] = factor(testDf[[f]])
    }
  }
  return(list(train=trainDf, test=testDf, componentColumns=componentColumns))
}

getCutOff = function(data) {
  fluctuate = data[data$y_cosine == 1,]
  nonFluctuate = data[data$y_cosine == 0,]
  
  flucFirstQ = quantile(fluctuate$y_predicted, 0.25)
  nonFlucFirstQ = quantile(nonFluctuate$y_predicted, 0.25)
  
  flucThirdQ = quantile(fluctuate$y_predicted, 0.75)
  nonFlucThirdQ = quantile(nonFluctuate$y_predicted, 0.75)
  
  if (flucFirstQ > nonFlucFirstQ & flucThirdQ > nonFlucThirdQ) {
    cutoff = abs(flucFirstQ + nonFlucThirdQ)/2
    return(cutoff)
  } else {
    cutoff = abs(median(fluctuate$y_predicted) + median(nonFluctuate$y_predicted))/2
    return(cutoff)
  }
}
