## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ----eval=FALSE---------------------------------------------------------------
## data(plpDataSimulationProfile)
## set.seed(1234)
## sampleSize <- 2000
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
##   addExposureDaysToStart = FALSE,
##   riskWindowEnd = 365,
##   addExposureDaysToEnd = FALSE,
##   verbosity = "INFO"
## )


## ----eval=FALSE---------------------------------------------------------------
## # Use LASSO logistic regression and Random Forest as base predictors
## model1 <- setLassoLogisticRegression()
## model2 <- setRandomForest()


## ----eval = FALSE-------------------------------------------------------------
## testFraction <- 0.2


## ----eval = FALSE-------------------------------------------------------------
## ensembleStrategy <- 'stacked'


## -----------------------------------------------------------------------------
# Use a split by person, alterantively a time split is possible
testSplit <- 'person'


## ----eval=FALSE---------------------------------------------------------------
## ensembleResults <- PatientLevelPrediction::runEnsembleModel(population,
##                                    dataList = list(plpData, plpData),
##                                    modelList = list(model1, model2),
##                                    testSplit=testSplit,
##                                    testFraction=testFraction,
##                                    nfold=3, splitSeed=1000,
##                                    ensembleStrategy = ensembleStrategy)


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## saveEnsemblePlpModel(ensembleResults$model, dirPath = file.path(getwd(),'model'))
## ensembleModel <- loadEnsemblePlpModel(getwd(),'model')


## ----eval=FALSE---------------------------------------------------------------
## plpData <- loadPlpData("<data file>")
## populationSettings <- ensembleModel$populationSettings
## populationSettings$plpData <- plpData
## population <- do.call(createStudyPopulation, populationSettings)


## ----eval=FALSE---------------------------------------------------------------
## ensembleModel <- loadEnsemblePlpModel("<model folder>")


## ----eval=FALSE---------------------------------------------------------------
## prediction <- applyEnsembleModel(population,
##                                   dataList = list(plpData, plpData),
##                                   ensembleModel = ensembleModel)$prediction


## ----eval=FALSE---------------------------------------------------------------
## # Show all demos in our package:
##  demo(package = "PatientLevelPrediction")
## 
## # Run the learning curve
##  demo("EnsembleModelDemo", package = "PatientLevelPrediction")


## ----tidy=TRUE,eval=TRUE------------------------------------------------------
citation("PatientLevelPrediction")

