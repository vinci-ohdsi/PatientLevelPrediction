## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ---- eval=FALSE, echo = FALSE, message = FALSE, warning = FALSE--------------
## library(PatientLevelPrediction)
## vignetteDataFolder <- "s:/temp/plpVignette"
## # Load all needed data if it exists on this computer:
## if (file.exists(vignetteDataFolder)){
##   plpModel <- loadPlpModel(vignetteDataFolder,'model')
##   lrResults <- loadPlpModel(file.path(vignetteDataFolder,'results'))
## }


## ----eval=FALSE---------------------------------------------------------------
## set.seed(1234)
## data(plpDataSimulationProfile)
## sampleSize <- 12000
## plpData <- simulatePlpData(
##   plpDataSimulationProfile,
##   n = sampleSize
## )
## 
## population <- createStudyPopulation(
##   plpData,
##   outcomeId = 2,
##   binary = TRUE,
##   firstExposureOnly = FALSE,
##   washoutPeriod = 0,
##   removeSubjectsWithPriorOutcome = FALSE,
##   priorOutcomeLookback = 99999,
##   requireTimeAtRisk = FALSE,
##   minTimeAtRisk = 0,
##   riskWindowStart = 0,
##   riskWindowEnd = 365,
##   verbosity = "INFO"
## )


## ----eval=FALSE---------------------------------------------------------------
## # Use LASSO logistic regression
## modelSettings <- setLassoLogisticRegression()
## 


## ----eval = FALSE-------------------------------------------------------------
## testFraction <- 0.2
## trainFractions <- seq(0.1, 0.8, 0.1) # Create eight training set fractions
## 
## # alternatively use a sequence of training events by uncommenting the line below.
## # trainEvents <- seq(100, 5000, 100)


## -----------------------------------------------------------------------------
# Use a split by person, alternatively a time split is possible
testSplit <- 'stratified'


## ----eval=FALSE---------------------------------------------------------------
## learningCurve <- createLearningCurve(population,
##                                      plpData = plpData,
##                                      modelSettings = modelSettings,
##                                      testFraction = 0.2,
##                                      verbosity = "TRACE",
##                                      trainFractions = trainFractions,
##                                      # trainEvents = trainEvents,
##                                      splitSeed = 1000,
##                                      saveModel = TRUE)


## ----eval=FALSE---------------------------------------------------------------
## plotLearningCurve(
##   learningCurve,
##   metric = 'AUROC',
##   abscissa = 'events',
##   plotTitle = 'Learning Curve',
##   plotSubtitle = 'AUROC performance'
## )


## ----eval=FALSE---------------------------------------------------------------
## learningCurvePar <- createLearningCurvePar(
##   population,
##   plpData =  plpData,
##   modelSettings = modelSettings,
##   testSplit = testSplit,
##   testFraction = testFraction,
##   trainEvents = trainEvents,
##   cores = 4,
##   splitSeed = 1000
## )


## ----eval=FALSE---------------------------------------------------------------
## # Show all demos in our package:
##  demo(package = "PatientLevelPrediction")
## 
## # Run the learning curve
##  demo("LearningCurveDemo", package = "PatientLevelPrediction")


## ----tidy=TRUE,eval=TRUE------------------------------------------------------
citation("PatientLevelPrediction")

