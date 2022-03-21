
infoChgList = c("DATALAB", "DM", "MESOS", "TDP", "TDQ", "TIMOB", "TMDM", "TUP", "XD")


writeCsv <- function(dataFrame, filePath) {
  print(paste0("write to ", filePath))
  write.csv(dataFrame, file = filePath ,row.names=FALSE, na="")
}


getRandomRoundCvRound <- function(roundCount){
  randomRound = as.integer(roundCount / 5)
  cvRound = roundCount %% 5
  return(c(randomRound, cvRound))
}


getNumberOfRound <- function(dataMode) {
  if (dataMode == 'cross') {
    return(49)
  } else {
    return(4)
  }
}

get_levels <- function(data,fac){
  levels(data[[fac]])
}
