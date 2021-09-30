## ---- echo = FALSE, message = FALSE, warning = FALSE--------------------------
library(PatientLevelPrediction)


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## setMadeUp <- function(a=1, b=2, seed=NULL){
##   # add input checks here...
## 
##   # now create list of all combinations:
##   result <- list(model='fitMadeUp', # this will be called to train the made up model
##                  param= split(expand.grid(a=a,
##                                           b=b,
##                                           seed=ifelse(is.null(seed),'NULL', seed)),
##                               1:(length(a)*length(b)  )),
##                  name='Made Up Algorithm'
##   )
##   class(result) <- 'modelSettings'
## 
##   return(result)
## }
## 


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## fitMadeUp <- function(population, plpData, param, quiet=F,
##                         outcomeId, cohortId, ...){
## 
##   # **************** code to train the model here
##   # trainedModel <- this code should apply each hyper-parameter using the cross validation
##   #                 then pick out the best hyper-parameter setting
##   #                 and finally fit a model on the whole train data using the
##   #                 optimal hyper-parameter settings
##   # ****************
## 
##   # construct the standard output for a model:
##   result <- list(model = trainedModel,
##                  modelSettings = list(model='made_up', modelParameters=param),
##                  trainCVAuc = NULL,
##                  hyperParamSearch = hyperSummary,
##                  metaData = plpData$metaData,
##                  populationSettings = attr(population, 'metaData'),
##                  outcomeId=outcomeId,# can use populationSettings$outcomeId?
##                  cohortId=cohortId,
##                  varImp = NULL,
##                  trainingTime=comp,
##                  covariateMap=result$map
##   )
##   class(result) <- 'plpModel'
##   attr(result, 'type') <- 'madeup'
##   attr(result, 'predictionType') <- 'binary'
##   return(result)
## 
## }
## 


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## predict.madeup <- function(plpModel,population, plpData, ...){
## 
##   # ************* code to do prediction for each rowId in population
##   # prediction <- code to do prediction here returning columns: rowId
##   #               and value (predicted risk)
##   #**************
## 
##   prediction <- merge(population, prediction, by='rowId')
##   prediction <- prediction[,colnames(prediction)%in%c('rowId','outcomeCount',
##                                                       'indexes', 'value')]
##   attr(prediction, "metaData") <- list(predictionType = "binary")
##   return(prediction)
## 
## }


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## setMadeUp <- function(a=1, b=2, seed=NULL){
##   # check a is valid positive value
##   if(missing(a)){
##     stop('a must be input')
##   }
##   if(!class(a)%in%c('numeric','integer'){
##     stop('a must be numeric')
##   }
##   if(a < 0){
##     stop('a must be positive')
##   }
##   # check b is numeric
##   if(missing(b)){
##     stop('b must be input')
##   }
##   if(!class(b)%in%c('numeric','integer'){
##     stop('b must be numeric')
##   }
## 
##   # now create list of all combinations:
##   result <- list(model='fitMadeUp',
##                  param= split(expand.grid(a=a,
##                                           b=b,
##                                           seed=ifelse(is.null(seed),'NULL', seed)),
##                               1:(length(a)*length(b)  )),
##                  name='Made Up Algorithm'
##   )
##   class(result) <- 'modelSettings'
## 
##   return(result)
## 
## 
## }
## 


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## fitMadeUp <- function(population, plpData, param, quiet=F,
##                         outcomeId, cohortId, ...){
##     if(!quiet)
##     writeLines('Training Made Up model')
## 
##   if(param[[1]]$seed!='NULL')
##     set.seed(param[[1]]$seed)
## 
##     # check plpData is coo format:
##   if(!'ffdf'%in%class(plpData$covariates) )
##     stop('This algorithm requires plpData in coo format')
## 
##   metaData <- attr(population, 'metaData')
##   if(!is.null(population$indexes))
##     population <- population[population$indexes>0,]
##   attr(population, 'metaData') <- metaData
## 
##   # convert data into sparse R Matrix:
##   result <- toSparseM(plpData,population,map=NULL)
##   data <- result$data
## 
##   data <- data[population$rowId,]
## 
##   # set test/train sets (for printing performance as it trains)
##   if(!quiet)
##     writeLines(paste0('Training made up model on train set containing ', nrow(population),
##                       ' people with ',sum(population$outcomeCount>0), ' outcomes'))
##   start <- Sys.time()
## 
##   #============= STEP 1 ======================================
##   # pick the best hyper-params and then do final training on all data...
##   writeLines('train')
##   datas <- list(population=population, data=data)
##   param.sel <- lapply(param, function(x) do.call(made_up_model, c(x,datas)  ))
##   hyperSummary <- do.call(rbind, lapply(param.sel, function(x) x$hyperSum))
##   hyperSummary <- as.data.frame(hyperSummary)
##   hyperSummary$auc <- unlist(lapply(param.sel, function(x) x$auc))
##   param.sel <- unlist(lapply(param.sel, function(x) x$auc))
##   param <- param[[which.max(param.sel)]]
## 
##   # set this so you do a final model train
##   param$final=T
## 
##   writeLines('final train')
##   trainedModel <- do.call(made_up_model, c(param,datas)  )$model
## 
##   comp <- Sys.time() - start
##   if(!quiet)
##     writeLines(paste0('Model Made Up trained - took:',  format(comp, digits=3)))
## 
##   # construct the standard output for a model:
##   result <- list(model = trainedModel,
##                  modelSettings = list(model='made_up', modelParameters=param),
##                  trainCVAuc = NULL,
##                  hyperParamSearch = hyperSummary,
##                  metaData = plpData$metaData,
##                  populationSettings = attr(population, 'metaData'),
##                  outcomeId=outcomeId,# can use populationSettings$outcomeId?
##                  cohortId=cohortId,
##                  varImp = NULL,
##                  trainingTime=comp,
##                  covariateMap=result$map
##   )
##   class(result) <- 'plpModel'
##   attr(result, 'type') <- 'madeup'
##   attr(result, 'predictionType') <- 'binary'
##   return(result)
## 
## }
## 


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## made_up_model <- function(data, population,
##                        a=1,b=1, final=F, ...){
## 
##   writeLines(paste('Training Made Up model with ',length(unique(population$indexes)),
##                    ' fold CV'))
##   if(!is.null(population$indexes) && final==F){
##     index_vect <- unique(population$indexes)
##     perform <- c()
## 
##     # create prediction matrix to store all predictions
##     predictionMat <- population
##     predictionMat$value <- 0
##     attr(predictionMat, "metaData") <- list(predictionType = "binary")
## 
##     for(index in 1:length(index_vect )){
##       writeLines(paste('Fold ',index, ' -- with ', sum(population$indexes!=index),
##                        'train rows'))
##       model <- madeup::model(x = data[population$indexes!=index,],
##                              y= population$outcomeCount[population$indexes!=index],
##                                   a=a, b=b)
## 
##       pred <- stats::predict(model, data[population$indexes==index,])
##       prediction <- population[population$indexes==index,]
##       prediction$value <- pred
##       attr(prediction, "metaData") <- list(predictionType = "binary")
##       aucVal <- computeAuc(prediction)
##       perform <- c(perform,aucVal)
## 
##       # add the fold predictions and compute AUC after loop
##       predictionMat$value[population$indexes==index] <- pred
## 
##      }
##     ##auc <- mean(perform) # want overal rather than mean
##     auc <- computeAuc(predictionMat)
## 
##     foldPerm <- perform
##   } else {
##     model <- madeup::model(x= data,
##                                 y= population$outcomeCount,
##                                 a=a,b=b)
## 
##     pred <- stats::predict(model, data)
##     prediction <- population
##     prediction$value <- pred
##     attr(prediction, "metaData") <- list(predictionType = "binary")
##     auc <- computeAuc(prediction)
##     foldPerm <- auc
##   }
## 
##   result <- list(model=model,
##                  auc=auc,
##                  hyperSum = unlist(list(a = a, b = b, fold_auc=foldPerm))
##   )
##   return(result)
## }


## ----tidy=FALSE,eval=FALSE----------------------------------------------------
## predict.madeup <- function(plpModel,population, plpData, ...){
##   result <- toSparseM(plpData, population, map=plpModel$covariateMap)
##   data <- result$data[population$rowId,]
##   prediction <- data.frame(rowId=population$rowId,
##                            value=stats::predict(plpModel$model, data)
##                            )
## 
##   prediction <- merge(population, prediction, by='rowId')
##   prediction <- prediction[,colnames(prediction)%in%
##                            c('rowId','outcomeCount','indexes', 'value')] # need to fix no index issue
##   attr(prediction, "metaData") <- list(predictionType = "binary")
##   return(prediction)
## 
## }


## ----tidy=TRUE,eval=TRUE------------------------------------------------------
citation("PatientLevelPrediction")

