## ----echo=FALSE,message=FALSE,warning=FALSE,eval=TRUE-------------------------
library(PatientLevelPrediction)
vignetteDataFolder <- "s:/temp/plpVignette"
# Load all needed data if it exists on this computer:
if (file.exists(vignetteDataFolder)){
plpModel <- loadPlpModel(vignetteDataFolder,'model')
lrResults <- loadPlpModel(file.path(vignetteDataFolder,'results'))
} 


## ---- echo = FALSE, message = FALSE, warning = FALSE--------------------------
library(PatientLevelPrediction)


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## library(PatientLevelPrediction)
## plpResult <- loadPlpResult('goodModel')
## 
## # add the model to the skeleton package with sensitive information removed
## exportPlpResult(plpResult = plpResult,
##                 modelName = 'Model Name',
##                 packageName = 'Your Package Name',
##                 gitHubLocation = 'location/of/github',
##                 includeEvaluationStatistics = T,
##                 includeThresholdSummary = T,
##                 includeDemographicSummary = T,
##                 includeCalibrationSummary = T,
##                 includePredictionDistribution = T,
##                 includeCovariateSummary = F)


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## library(PatientLevelPrediction)
## # input settings for person running the study
## connectionDetails <- ' '
## cdmDatabaseSchema <- 'their_cdm_database'
## databaseName <- 'Name for database'
## cohortDatabaseSchema <- 'a_database_with_write_priv'
## cohortTable <- 'package_table'
## outputLocation <- 'location to save results'
## 
## cohortDetails <- createCohort(connectionDetails = connectionDetails,
##                               cdmDatabaseSchema = cdmDatabaseSchema,
##                               cohortDatabaseSchema = cohortDatabaseSchema,
##                               cohortTable = cohortTable,
##              package = 'Your Package Name')
## 
## plpResult <- loadPlpResult(system.file("model",
##                           package = 'Your Package Name'))
## result <- externalValidatePlp(plpResult = plpResult,
##                               connectionDetails = connectionDetails,
##                     validationSchemaTarget = cohortDatabaseSchema,
##                     validationSchemaOutcome = cohortDatabaseSchema,
##                     validationSchemaCdm = cdmDatabaseSchema,
##                     validationTableTarget = cohortTable,
##                     validationTableOutcome = cohortTable,
##                     validationIdTarget = target_cohort_id,
##                     validationIdOutcome = outcome_cohort_id)
## 
## # save results to standard output
## resultLoc <- standardOutput(result = result,
##                outputLocation = outputLocation ,
##                studyName = 'external validation of ... model',
##                databaseName = databaseName,
##                cohortName = 'your cohortName',
##                outcomeName = 'your outcomeName')
## 
## # package results ready to submit
## packageResults(mainFolder=resultLoc,
##                includeROCplot= T,
##                includeCalibrationPlot = T,
##                includePRPlot = T,
##                includeTable1 = F,
##                includeThresholdSummary =T,
##                includeDemographicSummary = T,
##                includeCalibrationSummary = T,
##                includePredictionDistribution =T,
##                includeCovariateSummary = F,
##                removeLessThanN = F,
##                N = 10)
## 


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## submitResults(exportFolder=outputLocation,
##               dbName=databaseName, key, secret)
## 

