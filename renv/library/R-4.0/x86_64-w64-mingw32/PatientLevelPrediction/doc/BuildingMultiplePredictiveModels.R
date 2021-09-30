## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## # define all study population settings
## studyPop1 <- createStudyPopulationSettings(binary = T,
##                                           includeAllOutcomes = F,
##                                           removeSubjectsWithPriorOutcome = T,
##                                           priorOutcomeLookback = 99999,
##                                           requireTimeAtRisk = T,
##                                           minTimeAtRisk=364,
##                                           riskWindowStart = 1,
##                                           riskWindowEnd = 365,
##                                           verbosity = "INFO")
## 
## studyPop2 <- createStudyPopulationSettings(binary = T,
##                                            includeAllOutcomes = F,
##                                            removeSubjectsWithPriorOutcome = T,
##                                            priorOutcomeLookback = 99999,
##                                            requireTimeAtRisk = F,
##                                            minTimeAtRisk=364,
##                                            riskWindowStart = 1,
##                                            riskWindowEnd = 365,
##                                            verbosity = "INFO")
## 
## studyPop3 <- createStudyPopulationSettings(binary = T,
##                                            includeAllOutcomes = F,
##                                            removeSubjectsWithPriorOutcome = F,
##                                            priorOutcomeLookback = 99999,
##                                            requireTimeAtRisk = T,
##                                            minTimeAtRisk=364,
##                                            riskWindowStart = 1,
##                                            riskWindowEnd = 365,
##                                            verbosity = "INFO")
## 
## # combine these in a population setting list
## populationSettingList <- list(studyPop1,studyPop2,studyPop3)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## covSet1 <- createCovariateSettings(useDemographicsGender = T,
##                                    useDemographicsAgeGroup = T,
##                                    useConditionGroupEraAnyTimePrior = T,
##                                    useDrugGroupEraAnyTimePrior = T)
## 
## covSet2 <- createCovariateSettings(useDemographicsGender = T,
##                                    useDemographicsAgeGroup = T,
##                                    useConditionGroupEraAnyTimePrior = T,
##                                    useDrugGroupEraAnyTimePrior = F)
## 
## covariateSettingList <- list(covSet1, covSet2)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## gbm <- setGradientBoostingMachine()
## lr <- setLassoLogisticRegression()
## ada <- setAdaBoost()
## 
## modelList <- list(gbm, lr, ada)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## modelAnalysisList <- createPlpModelSettings(modelList = modelList,
##                                    covariateSettingList = covariateSettingList,
##                                    populationSettingList = populationSettingList)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## dbms <- "your dbms"
## user <- "your username"
## pw <- "your password"
## server <- "your server"
## port <- "your port"
## 
## connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = dbms,
##                                                                 server = server,
##                                                                 user = user,
##                                                                 password = pw,
##                                                                 port = port)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## cdmDatabaseSchema <- "your cdmDatabaseSchema"
## workDatabaseSchema <- "your workDatabaseSchema"
## cdmDatabaseName <- "your cdmDatabaseName"


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## allresults <- runPlpAnalyses(connectionDetails = connectionDetails,
##                            cdmDatabaseSchema = cdmDatabaseSchema,
##                            cdmDatabaseName = cdmDatabaseName,
##                            oracleTempSchema = cdmDatabaseSchema,
##                            cohortDatabaseSchema = workDatabaseSchema,
##                            cohortTable = "your cohort table",
##                            outcomeDatabaseSchema = workDatabaseSchema,
##                            outcomeTable = "your cohort table",
##                            cdmVersion = 5,
##                            outputFolder = "./PlpMultiOutput",
##                            modelAnalysisList = modelAnalysisList,
##                            cohortIds = c(2484,6970),
##                            cohortNames = c('visit 2010','test cohort'),
##                            outcomeIds = c(7331,5287),
##                            outcomeNames =  c('outcome 1','outcome 2'),
##                            maxSampleSize = NULL,
##                            minCovariateFraction = 0,
##                            normalizeData = T,
##                            testSplit = "stratified",
##                            testFraction = 0.25,
##                            splitSeed = NULL,
##                            nfold = 3,
##                            verbosity = "INFO")


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## val <- evaluateMultiplePlp(analysesLocation = "./PlpMultiOutput",
##                            outputLocation = "./PlpMultiOutput/validation",
##                            connectionDetails = connectionDetails,
##                            validationSchemaTarget = list('new_database_1.dbo',
##                                                               'new_database_2.dbo'),
##                            validationSchemaOutcome = list('new_database_1.dbo',
##                                                               'new_database_2.dbo'),
##                            validationSchemaCdm = list('new_database_1.dbo',
##                                                               'new_database_2.dbo'),
##                            databaseNames = c('database1','database2'),
##                            validationTableTarget = 'your new cohort table',
##                            validationTableOutcome = 'your new cohort table')


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## viewMultiplePlp(analysesLocation="./PlpMultiOutput")


## ----tidy=TRUE,eval=TRUE------------------------------------------------------
citation("PatientLevelPrediction")

