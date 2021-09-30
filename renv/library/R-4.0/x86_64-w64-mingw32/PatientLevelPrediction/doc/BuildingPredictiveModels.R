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


## ----table2, echo=FALSE, message=FALSE, warnings=FALSE, results='asis'--------
tabl <- "   
| Algorihm | Description | Hyper-parameters |
| ----------| ---------------------------------------------------| ----------------------- |
| Regularized Logistic Regression | Lasso logistic regression belongs to the family of generalized linear models, where a linear combination of the variables is learned and finally a logistic function maps the linear combination to a value between 0 and 1.  The lasso regularization adds a cost based on model complexity to the objective function when training the model.  This cost is the sum of the absolute values of the linear combination of the coefficients.  The model automatically performs feature selection by minimizing this cost. We use the Cyclic coordinate descent for logistic, Poisson and survival analysis (Cyclops) package to perform large-scale regularized logistic regression: https://github.com/OHDSI/Cyclops | var  (starting  variance), seed |
| Gradient boosting machines | Gradient boosting machines is a boosting ensemble technique and in our framework it combines multiple decision trees.  Boosting works by iteratively adding decision trees but adds more weight to the data-points that are misclassified by prior decision trees in the cost function when training the next tree.  We use Extreme Gradient Boosting, which is an efficient implementation of the gradient boosting framework implemented in the xgboost R package available from CRAN. | ntree (number of trees), max depth (max levels in tree), min rows (minimum data points in in node), learning rate, balance (balance class labels), seed  |
| Random forest | Random forest is a bagging ensemble technique that combines multiple decision trees.  The idea behind bagging is to reduce the likelihood of overfitting, by using weak classifiers, but combining multiple diverse weak classifiers into a strong classifier.  Random forest accomplishes this by training multiple decision trees but only using a subset of the variables in each tree and the subset of variables differ between trees. Our packages uses the sklearn learn implementation of Random Forest in python. | mtry  (number  of  features  in  each  tree),ntree (number of trees), maxDepth (max levels in tree), minRows (minimum data points in in node),balance (balance class labels), seed  |
| K-nearest neighbors | K-nearest neighbors (KNN) is an algorithm that uses some metric to find the K closest labelled data-points, given the specified metric, to a new unlabelled data-point.  The prediction of the new data-points is then the most prevalent class of the K-nearest labelled data-points.  There is a sharing limitation of KNN, as the model requires labelled data to perform the prediction on new data, and it is often not possible to share this data across data sites.We included the BigKnn classifier developed in OHDSI which is a large scale k-nearest neighbor classifier using the Lucene search engine: https://github.com/OHDSI/BigKnn | k (number   of   neighbours),weighted (weight by inverse frequency)  |
| Naive Bayes | The Naive Bayes algorithm applies the Bayes theorem with the 'naive' assumption of conditional independence between every pair of features given the value of the class variable. Based on the likelihood the data belongs to a class and the prior distribution of the class, a posterior distribution is obtained.  | none |
| AdaBoost | AdaBoost is a boosting ensemble technique. Boosting works by iteratively adding classifiers but adds more weight to the data-points that are misclassified by prior classifiers in the cost function when training the next classifier. We use the sklearn 'AdaboostClassifier' implementation in Python. | nEstimators (the maximum number of estimators at which boosting is terminated), learningRate (learning rate shrinks the contribution of each classifier by learning_rate. There is a trade-off between learningRate and nEstimators) |
| Decision Tree | A decision tree is a classifier that partitions the variable space using individual tests selected using a greedy approach.  It aims to find partitions that have the highest information gain to separate the classes.  The decision tree can easily overfit by enabling a large number of partitions (tree depth) and often needs some regularization (e.g., pruning or specifying hyper-parameters that limit the complexity of the model). We use the sklearn 'DecisionTreeClassifier' implementation in Python. | maxDepth (the maximum depth of the tree), minSamplesSplit,minSamplesLeaf, minImpuritySplit (threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.), seed,classWeight ('Balance' or 'None') |
| Multilayer Perception | Neural networks contain multiple layers that weight their inputs using a non-linear function.  The first layer is the input layer, the last layer is the output layer the between are the hidden layers.  Neural networks are generally trained using feed forward back-propagation.  This is when you go through the network with a data-point and calculate the error between the true label and predicted label, then go backwards through the network and update the linear function weights based on the error.  This can also be performed as a batch, where multiple data-points are fee| size (the number of hidden nodes), alpha (the l2 regularisation), seed |
| Deep Learning | Deep learning such as deep nets, convolutional neural networks or recurrent neural networks are similar to a neural network but have multiple hidden layers that aim to learn latent representations useful for prediction. In the seperate BuildingDeepLearningModels vignette we describe these models and hyper-parameters in more detail | see BuildingDeepLearningModels vignette  |
"
cat(tabl) # output the table in a format good for HTML/PDF/docx conversion


## /***********************************

## File AfStrokeCohorts.sql

## ***********************************/

## /*

## Create a table to store the persons in the T and C cohort

## */

## 
## IF OBJECT_ID('@resultsDatabaseSchema.PLPAFibStrokeCohort', 'U') IS NOT NULL

## DROP TABLE @resultsDatabaseSchema.PLPAFibStrokeCohort;

## 
## CREATE TABLE @resultsDatabaseSchema.PLPAFibStrokeCohort

## (

## cohort_definition_id INT,

## subject_id BIGINT,

## cohort_start_date DATE,

## cohort_end_date DATE

## );

## 
## 
## /*

## T cohort:  [PatientLevelPrediction vignette]:  T : patients who are newly

## diagnosed with Atrial fibrillation

## - persons with a condition occurrence record of 'Atrial fibrillation' or

## any descendants, indexed at the first diagnosis

## - who have >1095 days of prior observation before their first diagnosis

## - and have no warfarin exposure any time prior to first AFib diagnosis

## */

## INSERT INTO @resultsDatabaseSchema.AFibStrokeCohort (cohort_definition_id,

## subject_id,

## cohort_start_date,

## cohort_end_date)

## SELECT 1 AS cohort_definition_id,

## AFib.person_id AS subject_id,

## AFib.condition_start_date AS cohort_start_date,

## observation_period.observation_period_end_date AS cohort_end_date

## FROM

## (

##   SELECT person_id, min(condition_start_date) as condition_start_date

##   FROM @cdmDatabaseSchema.condition_occurrence

##   WHERE condition_concept_id IN (SELECT descendant_concept_id FROM

##   @cdmDatabaseSchema.concept_ancestor WHERE ancestor_concept_id IN

##   (313217 /*atrial fibrillation*/))

##   GROUP BY person_id

## ) AFib

##   INNER JOIN @cdmDatabaseSchema.observation_period

##   ON AFib.person_id = observation_period.person_id

##   AND AFib.condition_start_date >= dateadd(dd,1095,

##   observation_period.observation_period_start_date)

##   AND AFib.condition_start_date <= observation_period.observation_period_end_date

##   LEFT JOIN

##   (

##   SELECT person_id, min(drug_exposure_start_date) as drug_exposure_start_date

##   FROM @cdmDatabaseSchema.drug_exposure

##   WHERE drug_concept_id IN (SELECT descendant_concept_id FROM

##   @cdmDatabaseSchema.concept_ancestor WHERE ancestor_concept_id IN

##   (1310149 /*warfarin*/))

##   GROUP BY person_id

##   ) warfarin

##   ON Afib.person_id = warfarin.person_id

##   AND Afib.condition_start_date > warfarin.drug_exposure_start_date

##   WHERE warfarin.person_id IS NULL

##   ;

## 

##   /*

##   C cohort:  [PatientLevelPrediction vignette]:  O: Ischemic stroke events

##   - inpatient visits that include a condition occurrence record for

##   'cerebral infarction' and descendants, 'cerebral thrombosis',

##   'cerebral embolism', 'cerebral artery occlusion'

##   */

##   INSERT INTO @resultsDatabaseSchema.AFibStrokeCohort (cohort_definition_id,

##   subject_id,

##   cohort_start_date,

##   cohort_end_date)

##   SELECT 2 AS cohort_definition_id,

##   visit_occurrence.person_id AS subject_id,

##   visit_occurrence.visit_start_date AS cohort_start_date,

##   visit_occurrence.visit_end_date AS cohort_end_date

##   FROM

##   (

##   SELECT person_id, condition_start_date

##   FROM @cdmDatabaseSchema.condition_occurrence

##   WHERE condition_concept_id IN (SELECT DISTINCT descendant_concept_id FROM

##   @cdmDatabaseSchema.concept_ancestor WHERE ancestor_concept_id IN

##   (443454 /*cerebral infarction*/) OR descendant_concept_id IN

##   (441874 /*cerebral thrombosis*/, 375557 /*cerebral embolism*/,

##   372924 /*cerebral artery occlusion*/))

##   ) stroke

##   INNER JOIN @cdmDatabaseSchema.visit_occurrence

##   ON stroke.person_id = visit_occurrence.person_id

##   AND stroke.condition_start_date >= visit_occurrence.visit_start_date

##   AND stroke.condition_start_date <= visit_occurrence.visit_end_date

##   AND visit_occurrence.visit_concept_id IN (9201, 262 /*'Inpatient Visit'  or

##   'Emergency Room and Inpatient Visit'*/)

##   GROUP BY visit_occurrence.person_id, visit_occurrence.visit_start_date,

##   visit_occurrence.visit_end_date

##   ;

## 


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##   connectionDetails <- createConnectionDetails(dbms = "postgresql",
##   server = "localhost/ohdsi",
##   user = "joe",
##   password = "supersecret")
## 
##   cdmDatabaseSchema <- "my_cdm_data"
##   cohortsDatabaseSchema <- "my_results"
##   cdmVersion <- "5"


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##   library(SqlRender)
##   sql <- readSql("AfStrokeCohorts.sql")
##   sql <- renderSql(sql,
##   cdmDatabaseSchema = cdmDatabaseSchema,
##   cohortsDatabaseSchema = cohortsDatabaseSchema,
##   post_time = 30,
##   pre_time = 365)$sql
##   sql <- translateSql(sql, targetDialect = connectionDetails$dbms)$sql
## 
##   connection <- connect(connectionDetails)
##   executeSql(connection, sql)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##   sql <- paste("SELECT cohort_definition_id, COUNT(*) AS count",
##   "FROM @cohortsDatabaseSchema.AFibStrokeCohort",
##   "GROUP BY cohort_definition_id")
##   sql <- renderSql(sql, cohortsDatabaseSchema = cohortsDatabaseSchema)$sql
##   sql <- translateSql(sql, targetDialect = connectionDetails$dbms)$sql
## 
##   querySql(connection, sql)

## ----echo=FALSE,message=FALSE-------------------------------------------------
data.frame(cohort_definition_id = c(1, 2),count = c(527616, 221555))


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##   covariateSettings <- createCovariateSettings(useDemographicsGender = TRUE,
##   useDemographicsAge = TRUE,
##   useConditionGroupEraLongTerm = TRUE,
##   useConditionGroupEraAnyTimePrior = TRUE,
##   useDrugGroupEraLongTerm = TRUE,
##   useDrugGroupEraAnyTimePrior = TRUE,
##   useVisitConceptCountLongTerm = TRUE,
##   longTermStartDays = -365,
##   endDays = -1)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##   plpData <- getPlpData(connectionDetails = connectionDetails,
##   cdmDatabaseSchema = cdmDatabaseSchema,
##   cohortDatabaseSchema = resultsDatabaseSchema,
##   cohortTable = 'AFibStrokeCohort',
##   cohortId = 1,
##   covariateSettings = covariateSettings,
##   outcomeDatabaseSchema = resultsDatabaseSchema,
##   outcomeTable = 'AFibStrokeCohort',
##   outcomeIds = 2,
##   sampleSize = 10000
##   )


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##   savePlpData(plpData, "stroke_in_af_data")


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##   population <- createStudyPopulation(plpData = plpData,
##   outcomeId = 2,
##   washoutPeriod = 1095,
##   firstExposureOnly = FALSE,
##   removeSubjectsWithPriorOutcome = FALSE,
##   priorOutcomeLookback = 1,
##   riskWindowStart = 1,
##   riskWindowEnd = 365,
##   addExposureDaysToStart = FALSE,
##   addExposureDaysToEnd = FALSE,
##   minTimeAtRisk = 364,
##   requireTimeAtRisk = TRUE,
##   includeAllOutcomes = TRUE,
##   verbosity = "DEBUG"
##   )


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## lrModel <- setLassoLogisticRegression()


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##   lrResults <- runPlp(population, plpData, modelSettings = lrModel, testSplit='stratified',
##   testFraction=0.25, nfold=2, splitSeed = 1234)


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## savePlpModel(lrResults$model, dirPath = file.path(getwd(), "model"))


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##   plpModel <- loadPlpModel(file.path(getwd(),'model'))


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##   savePlpResult(lrResults, location = file.path(getwd(),'lr'))


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##   lrResults <- loadPlpResult(file.path(getwd(),'lr'))


##   /***********************************

##     File AceAngioCohorts.sql

##   ***********************************/

##     /*

##     Create a table to store the persons in the T and C cohort

##   */

## 

##     IF OBJECT_ID('@resultsDatabaseSchema.PLPAceAngioCohort', 'U') IS NOT NULL

##   DROP TABLE @resultsDatabaseSchema.PLPAceAngioCohort;

## 

##   CREATE TABLE @resultsDatabaseSchema.PLPAceAngioCohort

##   (

##     cohort_definition_id INT,

##     subject_id BIGINT,

##     cohort_start_date DATE,

##     cohort_end_date DATE

##   );

## 

## 

##   /*

##     T cohort:  [PatientLevelPrediction vignette]:  T : patients who are newly

##   dispensed an ACE inhibitor

##   - persons with a drug exposure record of any 'ACE inhibitor' or

##   any descendants, indexed at the first diagnosis

##   - who have >364 days of prior observation before their first dispensing

##   */

##     INSERT INTO @resultsDatabaseSchema.AceAngioCohort (cohort_definition_id,

##                                                        subject_id,

##                                                        cohort_start_date,

##                                                        cohort_end_date)

##   SELECT 1 AS cohort_definition_id,

##   Ace.person_id AS subject_id,

##   Ace.drug_start_date AS cohort_start_date,

##   observation_period.observation_period_end_date AS cohort_end_date

##   FROM

##   (

##     SELECT person_id, min(drug_exposure_date) as drug_start_date

##     FROM @cdmDatabaseSchema.drug_exposure

##     WHERE drug_concept_id IN (SELECT descendant_concept_id FROM

##                               @cdmDatabaseSchema.concept_ancestor WHERE ancestor_concept_id IN

##                               (1342439,1334456, 1331235, 1373225, 1310756, 1308216, 1363749, 1341927, 1340128, 1335471 /*ace inhibitors*/))

##     GROUP BY person_id

##   ) Ace

##   INNER JOIN @cdmDatabaseSchema.observation_period

##   ON Ace.person_id = observation_period.person_id

##   AND Ace.drug_start_date >= dateadd(dd,364,

##                                      observation_period.observation_period_start_date)

##   AND Ace.drug_start_date <= observation_period.observation_period_end_date

##   ;

## 

##   /*

##     C cohort:  [PatientLevelPrediction vignette]:  O: Angioedema

##   */

##     INSERT INTO @resultsDatabaseSchema.AceAngioCohort (cohort_definition_id,

##                                                        subject_id,

##                                                        cohort_start_date,

##                                                        cohort_end_date)

##   SELECT 2 AS cohort_definition_id,

##   angioedema.person_id AS subject_id,

##   angioedema.condition_start_date AS cohort_start_date,

##   angioedema.condition_start_date AS cohort_end_date

##   FROM

##   (

##     SELECT person_id, condition_start_date

##     FROM @cdmDatabaseSchema.condition_occurrence

##     WHERE condition_concept_id IN (SELECT DISTINCT descendant_concept_id FROM

##                                    @cdmDatabaseSchema.concept_ancestor WHERE ancestor_concept_id IN

##                                    (432791 /*angioedema*/) OR descendant_concept_id IN

##                                    (432791 /*angioedema*/)

##     ) angioedema

## 

##     ;

## 


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##     connectionDetails <- createConnectionDetails(dbms = "postgresql",
##                                                  server = "localhost/ohdsi",
##                                                  user = "joe",
##                                                  password = "supersecret")
## 
##     cdmDatabaseSchema <- "my_cdm_data"
##     cohortsDatabaseSchema <- "my_results"
##     cdmVersion <- "5"


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##     library(SqlRender)
##     sql <- readSql("AceAngioCohorts.sql")
##     sql <- render(sql,
##                   cdmDatabaseSchema = cdmDatabaseSchema,
##                   cohortsDatabaseSchema = cohortsDatabaseSchema)
##     sql <- translate(sql, targetDialect = connectionDetails$dbms)
## 
##     connection <- connect(connectionDetails)
##     executeSql(connection, sql)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##     sql <- paste("SELECT cohort_definition_id, COUNT(*) AS count",
##                  "FROM @cohortsDatabaseSchema.AceAngioCohort",
##                  "GROUP BY cohort_definition_id")
##     sql <- render(sql, cohortsDatabaseSchema = cohortsDatabaseSchema)
##     sql <- translate(sql, targetDialect = connectionDetails$dbms)
## 
##     querySql(connection, sql)

## ----echo=FALSE,message=FALSE-------------------------------------------------
    data.frame(cohort_definition_id = c(1, 2),count = c(0, 0))


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##     covariateSettings <- createCovariateSettings(useDemographicsGender = TRUE,
##                                                  useDemographicsAge = TRUE,
##                                                  useConditionGroupEraLongTerm = TRUE,
##                                                  useConditionGroupEraAnyTimePrior = TRUE,
##                                                  useDrugGroupEraLongTerm = TRUE,
##                                                  useDrugGroupEraAnyTimePrior = TRUE,
##                                                  useVisitConceptCountLongTerm = TRUE,
##                                                  longTermStartDays = -365,
##                                                  endDays = -1)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##     plpData <- getPlpData(connectionDetails = connectionDetails,
##                           cdmDatabaseSchema = cdmDatabaseSchema,
##                           cohortDatabaseSchema = resultsDatabaseSchema,
##                           cohortTable = 'AceAngioCohort',
##                           cohortId = 1,
##                           covariateSettings = covariateSettings,
##                           outcomeDatabaseSchema = resultsDatabaseSchema,
##                           outcomeTable = 'AceAngioCohort',
##                           outcomeIds = 2,
##                           sampleSize = 10000
##     )


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##     savePlpData(plpData, "angio_in_ace_data")


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##     population <- createStudyPopulation(plpData = plpData,
##                                         outcomeId = 2,
##                                         washoutPeriod = 364,
##                                         firstExposureOnly = FALSE,
##                                         removeSubjectsWithPriorOutcome = TRUE,
##                                         priorOutcomeLookback = 9999,
##                                         riskWindowStart = 1,
##                                         riskWindowEnd = 365,
##                                         addExposureDaysToStart = FALSE,
##                                         addExposureDaysToEnd = FALSE,
##                                         minTimeAtRisk = 364,
##                                         requireTimeAtRisk = TRUE,
##                                         includeAllOutcomes = TRUE,
##                                         verbosity = "DEBUG"
##     )


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##     gbmModel <- setGradientBoostingMachine(ntrees = 5000, maxDepth = c(4,7,10), learnRate = c(0.001,0.01,0.1,0.9))


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
##     gbmResults <- runPlp(population, plpData, modelSettings = gbmModel, testSplit='stratified',
##                          testFraction=0.25, nfold=2, splitSeed = 1234)


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##     savePlpModel(gbmResults$model, dirPath = file.path(getwd(), "model"))


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##     plpModel <- loadPlpModel(file.path(getwd(),'model'))


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##     savePlpResult(gbmResults, location = file.path(getwd(),'gbm'))


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
##     gbmResults <- loadPlpResult(file.path(getwd(),'gbm'))


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## plotPlp(lrResults, dirPath=getwd())


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## plotSmoothCalibration(lrResults)


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## P <- Tp/(Tp+Fp)


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## R <- Tp/(Tp + Fn)


## ----tidy=TRUE,eval=FALSE-----------------------------------------------------
## F1 <- 2*P*R/(P+R)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## # load the trained model
## plpModel <- loadPlpModel(getwd(),'model')
## 
## #load the new plpData and create the population
## plpData <- loadPlpData(getwd(),'data')
## population <- createStudyPopulation(plpData,
##                                     outcomeId = 2,
##                                     includeAllOutcomes = TRUE,
##                                     firstExposureOnly = TRUE,
##                                     washoutPeriod = 365,
##                                     removeSubjectsWithPriorOutcome = TRUE,
##                                     priorOutcomeLookback = 365,
##                                     riskWindowStart = 1,
##                                     requireTimeAtRisk = FALSE,
##                                     riskWindowEnd = 365
## )
## 
## # apply the trained model on the new data
## validationResults <- applyModel(population,plpData,plpModel)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## # load the trained model
## plpResult <- loadPlpResult(getwd(),'plpResult')
## 
## connectionDetails <- createConnectionDetails(dbms = "postgresql",
##                                              server = "localhost/ohdsi",
##                                              user = "joe",
##                                              password = "supersecret")
## 
## validation <- externalValidatePlp(plpResult = plpResult,
##                                   connectionDetails = connectionDetails,
##                                   validationSchemaTarget = 'new_cohort_schema',
##                                   validationSchemaOutcome = 'new_cohort_schema',
##                                   validationSchemaCdm = 'new_cdm_schema',
##                                   validationTableTarget = 'cohort_table',
##                                   validationTableOutcome = 'cohort_table',
##                                   validationIdTarget = 'cohort_id',
##                                   validationIdOutcome = 'outcome_id',
##                                   keepPrediction = T
## )
## 


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## # load the trained model
## plpResult <- loadPlpResult(getwd(),'plpResult')
## 
## connectionDetails <- createConnectionDetails(dbms = "postgresql",
##                                              server = "localhost/ohdsi",
##                                              user = "joe",
##                                              password = "supersecret")
## 
## validation <- externalValidatePlp(plpResult = plpResult,
##                                   connectionDetails = connectionDetails,
##                                   validationSchemaTarget = list('new_cohort_schema1',
##                                                                 'new_cohort_schema2'),
##                                   validationSchemaOutcome = list('new_cohort_schema1',
##                                                                  'new_cohort_schema2'),
##                                   validationSchemaCdm = list('new_cdm_schema1',
##                                                              'new_cdm_schema2'),
##                                   validationTableTarget = list('new_cohort_table1',
##                                                                'new_cohort_table2'),
##                                   validationTableOutcome = list('new_cohort_table1',
##                                                                 'new_cohort_table2'),
##                                   validationIdTarget = 'cohort_id',
##                                   validationIdOutcome = 'outcome_id',
##                                   keepPrediction = T
## )
## 


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## 
## createPlpJournalDocument(plpResult = <your plp results>,
##                          plpValidation = <your validation results>,
##                          plpData = <your plp data>,
##                          targetName = "<target population>",
##                          outcomeName = "<outcome>",
##                          table1 = F,
##                          connectionDetails = NULL,
##                          includeTrain = FALSE,
##                          includeTest = TRUE,
##                          includePredictionPicture = TRUE,
##                          includeAttritionPlot = TRUE,
##                          outputLocation = "<your location>"))
## ````
## 
## For more details see the help page of the function.
## 
## \newpage
## # Other functionality
## 
## The package has much more functionality than described in this vignette and contributions have been made my many persons in the OHDSI community. The table below provides an overview:
## 
##   Functionality | Description | Vignette
## --------------- | --------------------------------------------- | ---------------
##   Builing Multiple Models | This vignette describes how you can run multiple models automatically | [`Vignette`](https://github.com/OHDSI/PatientLevelPrediction/blob/master/inst/doc/BuildingMultiplePredictiveModels.pdf)
## Custom algorithms | This vignette describes how you can add your own custom algorithms in the framework | [`Vignette`](https://github.com/OHDSI/PatientLevelPrediction/blob/master/inst/doc/AddingCustomAlgorithms.pdf)
## Ensemble models | This vignette describes how you can use the framework to build ensemble models, i.e combine multiple models in a super learner | [`Vignette`](https://github.com/OHDSI/PatientLevelPrediction/blob/master/inst/doc/BuildingEnsembleModels.pdf)
## Deep Learning Models | We have added extensive functionality for Deep Learning including several architectures in both pyTorch and Keras. These algorithms can be trained using GPU power | [`Vignette`](https://github.com/OHDSI/PatientLevelPrediction/blob/master/inst/doc/BuildingDeepLearningModels.pdf)
## Learning curves | Learning curves assess the effect of training set size on model performance by training a sequence of prediction models on successively larger subsets of the training set. A learning curve plot can also help in diagnosing a bias or variance problem as explained below. | [`Vignette`](https://github.com/OHDSI/PatientLevelPrediction/blob/master/inst/doc/GeneratingLearningCurves.pdf)
## Implementing existing models | This vignette describes how you can implement existing logistic regression models in the framework, e.g. as found in literature | [`Vignette`](https://github.com/OHDSI/PatientLevelPrediction/blob/master/inst/doc/ImplementingExistingModels.pdf)
## 
## # Demos
## 
## We have added several demos in the package that run on simulated data:
## 

## ----eval=FALSE---------------------------------------------------------------
## # Show all demos in our package:
## demo(package = "PatientLevelPrediction")
## 
## # For example, to run the SingleModelDemo that runs Lasso and shows you how to run the Shiny App use this call
## demo("SingleModelDemo", package = "PatientLevelPrediction")


## ----tidy=TRUE,eval=TRUE------------------------------------------------------
citation("PatientLevelPrediction")


## ----tidy=TRUE,eval=TRUE------------------------------------------------------
citation("Cyclops")

