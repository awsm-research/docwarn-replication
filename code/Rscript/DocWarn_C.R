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
buildModels <- function(dataMode, project, i, selectedColumns) {
  print(paste("build ", dataMode, project, i))
  r = getRandomRoundCvRound(i)
  randomRound = r[1]
  cvRound = r[2]
  
  model_formula = formula(paste("y_cosine" , " ~ " , paste(selectedColumns,collapse="+")))
  
  ########## Get trainDf for random forest (ensure factor levels are similar for both train and test)
  tempList = getDf(dataMode, project, randomRound, cvRound, forRandomForest = TRUE, model_formula=model_formula)
  trainDf = tempList$train
  dd <<- datadist(trainDf[selectedColumns])
  options(datadist="dd") 
  
  rf = randomForest(model_formula, data=trainDf, importance=TRUE, type = "classification", ntree=300)
  save(rf, file = paste0("../../data/modelResult/models/", dataMode, "/", project, "_rf_", randomRound, "_", cvRound, ".rda"))
}


for (dataMode in c('time', 'cross')){
  for (project in infoChgList) {
    # get columns to be used for this project
    survivedColumns = c()
    for (i in 0:getNumberOfRound(dataMode)) {
      print(paste("getcolumns ", dataMode, project, i))
      r = getRandomRoundCvRound(i)
      randomRound = r[1]
      cvRound = r[2]
      
      ##########
      tempList = getDf(dataMode, project, randomRound, cvRound, forRandomForest=FALSE, model_formula=model_formula)
      trainDf = tempList$train
      trainDf = trainDf[sapply(trainDf, function(x) length(unique(na.omit(x)))) > 1]
      componentColumns = tempList$componentColumns
      numericColumns = colnames(trainDf)
      numericColumns = numericColumns[which(!numericColumns %in% c('issue_key', 'issuetype', 'type', 'y_cosine'))]
      numericColumns = numericColumns[which(!numericColumns %in% factorFeatures)]
      numericColumns = numericColumns[which(!numericColumns %in% componentColumns)]
      
      cols = removeVars_new(trainDf[numericColumns], numericColumns)
      survivedColumns = c(survivedColumns, cols)
      
      nonConstantColumns = colnames(trainDf)
      survivedColumns = c(survivedColumns, componentColumns[which(componentColumns %in% nonConstantColumns)])
      survivedColumns = c(survivedColumns, factorFeatures[factorFeatures %in% nonConstantColumns])
    }
    tempTable = table(survivedColumns)
    highestCount = max(tempTable)
    selectedColumns = names(tempTable[tempTable == highestCount])
    
    # build models based on the selected columns
    for (i in 0:getNumberOfRound(dataMode)) {
      buildModels(dataMode, project, i, selectedColumns)
    }
  }
}


