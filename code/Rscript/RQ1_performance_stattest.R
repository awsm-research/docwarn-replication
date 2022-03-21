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
source("RQ4_util.R")
########################################################################################


library(plyr)
library(dplyr)
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
library(effsize)
library(coin)

getEffSize = function(z, n) {
  effSize = z/sqrt(n)
  if (effSize >= 0.8) {
    return("L")
  } else if (effSize >= 0.5) {
    return("M")
  } else if (effSize >= 0.2) {
    return("S")
  } else {
    return("N")
  }
}

translatePValue = function(p) {
  if (p < 0.001) {
    return("***")
  } else if (p < 0.01) {
    return("**")
  } else if (p < 0.05) {
    return("*")
  } else {
    return("◦")
  }
}

compare = function(main, comparing) {
  wilcoxResult = wilcoxsign_test(main ~ comparing, distribution="exact", alternative="greater")
  effectSize = getEffSize(attr(wilcoxResult@statistic, "teststatistic"), 50)
  pvalue = pvalue(wilcoxResult)[1]
  return(paste0(effectSize, translatePValue(pvalue)))
}


robertaDf = read.csv(paste0("../../data/modelResult/roberta_cross_all.csv"))
robertaFeatureDf = read.csv(paste0("../../data/modelResult/roberta_features_cross_all.csv"))


##### Compare each variation of PDC with the baselines


resultDf = data.frame()
for (projectKey in infoChgList) {
  feature = read.csv(paste0("../../data/modelResult/performance_features/", projectKey, "_cross_all.csv"))
  rf_auc = feature$rf_auc
  rf_f1 = feature$rf_f
  
  #load random
  random_auc = feature$random_auc
  random_f1 = feature$random_f
  #load oner
  oner_auc = feature$oner_auc
  oner_f1 = feature$oner_f
  
  #compare rfFeature with random and OneR
  rf_random_auc_result = compare(rf_auc, random_auc)
  rf_random_f1_result = compare(rf_f1, random_f1)
  rf_oner_auc_result = compare(rf_auc, oner_auc)
  rf_oner_f1_result = compare(rf_f1, oner_f1)
  
  #load roberta
  robertaProjDf = robertaDf[robertaDf$project == projectKey,]
  roberta_auc = robertaProjDf$auc
  roberta_f1 = robertaProjDf$f1
  #compare roberta text with random and oner
  roberta_random_auc_result = compare(roberta_auc, random_auc)
  roberta_random_f1_result = compare(roberta_f1, random_f1)
  roberta_oner_auc_result = compare(roberta_auc, oner_auc)
  roberta_oner_f1_result = compare(roberta_f1, oner_f1)
  
  #load roberta features
  robertaFeaturesProjDf = robertaFeatureDf[robertaFeatureDf$project == projectKey,]
  robertaFeature_auc = robertaFeaturesProjDf$auc
  robertaFeature_f1 = robertaFeaturesProjDf$f1
  #compare roberta features with random and oner
  robertaFeature_random_auc_result = compare(robertaFeature_auc, random_auc)
  robertaFeature_random_f1_result = compare(robertaFeature_f1, random_f1)
  robertaFeature_oner_auc_result = compare(robertaFeature_auc, oner_auc)
  robertaFeature_oner_f1_result = compare(robertaFeature_f1, oner_f1)
  
  resultRow = data.frame('project'=projectKey,
                         'rf_random_auc_result'=rf_random_auc_result, 
                         'rf_random_f1_result'=rf_random_f1_result, 
                         'rf_oner_auc_result'=rf_oner_auc_result, 
                         'rf_oner_f1_result'=rf_oner_f1_result,
                         
                         'roberta_random_auc_result'=roberta_random_auc_result, 
                         'roberta_random_f1_result'=roberta_random_f1_result,
                         'roberta_oner_auc_result'=roberta_oner_auc_result, 
                         'roberta_oner_f1_result'=roberta_oner_f1_result,
                         
                         'robertaFeature_random_auc_result'=robertaFeature_random_auc_result, 
                         'robertaFeature_random_f1_result'=robertaFeature_random_f1_result,
                         'robertaFeature_oner_auc_result'=robertaFeature_oner_auc_result, 
                         'robertaFeature_oner_f1_result'=robertaFeature_oner_f1_result
  )  
  resultDf = rbind(resultDf, resultRow)
}

writeCsv(dataFrame = resultDf, "../../data/modelResult/stattest_pdc_and_baseline.csv")




#### COMPARE PDC Features and Text / Hybrid and baselines (i.e., OneR and random)
resultDf = data.frame()
for (projectKey in infoChgList) {
  feature = read.csv(paste0("../../data/modelResult/performance_features/", projectKey, "_cross_all.csv"))
  rf_auc = feature$rf_auc
  rf_f1 = feature$rf_f
  
  #load random
  random_auc = feature$random_auc
  random_f1 = feature$random_f
  #compare rfFeature with random
  random_auc = compare(rf_auc, random_auc)
  random_f1 = compare(rf_f1, random_f1)
  
  
  #load oner
  oner_auc = feature$oner_auc
  oner_f1 = feature$oner_f
  #compare rfFeature with oner
  oner_auc = compare(rf_auc, oner_auc)
  oner_f1 = compare(rf_f1, oner_f1)
  
  
  
  #load roberta
  robertaProjDf = robertaDf[robertaDf$project == projectKey,]
  roberta_auc = robertaProjDf$auc
  roberta_f1 = robertaProjDf$f1
  #compare rfFeature with roberta text
  roberta_auc = compare(rf_auc, roberta_auc)
  roberta_f1 = compare(rf_f1, roberta_f1)
  
  #load roberta features
  robertaFeaturesProjDf = robertaFeatureDf[robertaFeatureDf$project == projectKey,]
  robertaFeature_auc = robertaFeaturesProjDf$auc
  robertaFeature_f1 = robertaFeaturesProjDf$f1
  #compare rfFeature with roberta features
  robertaFeature_auc = compare(rf_auc, robertaFeature_auc)
  robertaFeature_f1 = compare(rf_f1, robertaFeature_f1)
  
  resultRow = data.frame('project'=projectKey,
                         'roberta_auc'=roberta_auc, 
                         'roberta_f1'=roberta_f1, 
                         'robertaFeature_auc'=robertaFeature_auc, 
                         'robertaFeature_f1'=robertaFeature_f1,
                         'random_auc'=random_auc, 
                         'random_f1'=random_f1,
                         'oner_auc'=oner_auc, 
                         'oner_f1'=oner_f1
  )  
  resultDf = rbind(resultDf, resultRow)
  
}

writeCsv(dataFrame = resultDf, "../../data/modelResult/stattest_pdcfeatures_and_others.csv")


