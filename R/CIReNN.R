# @file CIReNN.R
#
# Copyright 2017 Observational Health Data Sciences and Informatics
#
# This file is part of PatientLevelPrediction
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



library(keras)
# restricts to pop and saves/creates mapping
MapCovariates <- function(covariates, covariateRef, population, map){

  # restrict to population for speed
  futile.logger::flog.trace('restricting to population for speed...')
  idx <- ffbase::ffmatch(x = covariates$rowId, table = ff::as.ff(population$rowId))
  idx <- ffbase::ffwhich(idx, !is.na(idx))
  covariates <- covariates[idx, ]

  futile.logger::flog.trace('Now converting covariateId...')
  oldIds <- as.double(ff::as.ram(covariateRef$covariateId))
  newIds <- 1:nrow(covariateRef)

  if(!is.null(map)){
    futile.logger::flog.trace('restricting to model variables...')
    futile.logger::flog.trace(paste0('oldIds: ',length(map[,'oldIds'])))
    futile.logger::flog.trace(paste0('newIds:', max(as.double(map[,'newIds']))))
    ind <- ffbase::ffmatch(x=covariateRef$covariateId, table=ff::as.ff(as.double(map[,'oldIds'])))
    ind <- ffbase::ffwhich(ind, !is.na(ind))
    covariateRef <- covariateRef[ind,]

    ind <- ffbase::ffmatch(x=covariates$covariateId, table=ff::as.ff(as.double(map[,'oldIds'])))
    ind <- ffbase::ffwhich(ind, !is.na(ind))
    covariates <- covariates[ind,]
  }
  if(is.null(map))
    map <- data.frame(oldIds=oldIds, newIds=newIds)

  return(list(covariates=covariates,
              covariateRef=covariateRef,
              map=map))
}


 toSparseArray <- function(plpData,population, map=NULL
                           #,minCovariateFraction = 0.01
                          ){
  # if (minCovariateFraction != 0){
  #    tidyCovariates<-FeatureExtraction::tidyCovariateData(covariates=plpData$covariates, 
  #                                              covariateRef=plpData$covariateRef,
  #                                              populationSize = nrow(plpData$cohorts),
  #                                              minFraction = minCovariateFraction,
  #                                              normalize = TRUE,
  #                                              removeRedundancy = FALSE)
  #    cov <- ff::clone(tidyCovariates$covariates)
  #    covref <- ff::clone(tidyCovariates$covariateRef)
  #  }else{
   #   cov <- ff::clone(plpData$covariates)
   #   covref <- ff::clone(plpData$covariateRef)
   # }
  cov <- ff::clone(plpData$covariates)
  covref <- ff::clone(plpData$covariateRef)
   
  plpData.mapped <- MapCovariates(covariates=cov, covariateRef=ff::clone(plpData$covariateRef),
                                  population, map)
  
  for (i in bit::chunk(plpData.mapped$covariateRef$covariateId)) {
    ids <- plpData.mapped$covariateRef$covariateId[i[1]:i[2]]
    ids <- plyr::mapvalues(ids, as.double(plpData.mapped$map$oldIds), as.double(plpData.mapped$map$newIds), warn_missing = FALSE)
    plpData.mapped$covariateRef$covariateId[i[1]:i[2]] <- ids
    # tested and working
  }
  for (i in bit::chunk(plpData.mapped$covariates$covariateId)) {
    ids <- plpData.mapped$covariates$covariateId[i[1]:i[2]]
    ids <- plyr::mapvalues(ids, as.double(plpData.mapped$map$oldIds), as.double(plpData.mapped$map$newIds), warn_missing = FALSE)
    plpData.mapped$covariates$covariateId[i[1]:i[2]] <- ids
  }
  futile.logger::flog.debug(paste0('Max ',ffbase::max.ff(plpData.mapped$covariates$covariateId)))
  
  #convert into sparseM
  futile.logger::flog.debug(paste0('# cols: ', nrow(plpData.mapped$covariateRef)))
  futile.logger::flog.debug(paste0('Max rowId: ', ffbase::max.ff(plpData.mapped$covariates$rowId)))
  
  # chunk then add
  for(i in min(cov$timeId):max(cov$timeId)){
    plpData.mapped$temp_covariates<-plpData.mapped$covariates[plpData.mapped$covariates$timeId==i]
    data <- Matrix::sparseMatrix(i=1,
                                 j=1,
                                 x=0,
                                 dims=c(max(population$rowId), max(plpData.mapped$map$newIds))) # edit this to max(map$newIds)
    for (ind in bit::chunk(plpData.mapped$temp_covariates$covariateId)) {
      futile.logger::flog.debug(paste0('start:', ind[1],'- end:',ind[2]))
      temp <- futile.logger::ftry(Matrix::sparseMatrix(i=ff::as.ram(plpData.mapped$temp_covariates$rowId[ind]),
                                                       j=ff::as.ram(plpData.mapped$temp_covariates$covariateId[ind]),
                                                       x=ff::as.ram(plpData.mapped$temp_covariates$covariateValue[ind]),
                                                       dims=c(max(population$rowId), max(plpData.mapped$map$newIds)))
      )
      data <- data+temp
    }
      data_array<-slam::as.simple_sparse_array(data)
      #extending one more dimesion to the array
      data_array<-slam::extend_simple_sparse_array(data_array,c(1L))
      #binding arrays along the dimesion
      if(i==min(cov$timeId)) {result_array<-data_array
      }else{
        result_array<-slam::abind_simple_sparse_array(result_array,data_array,MARGIN=2L)
      }
  }
   futile.logger::flog.debug(paste0('Sparse array with dimensionality: ', paste(dim(data), collapse=',')  ))
  
  result <- list(data=result_array,
                 covariateRef=plpData.mapped$covariateRef,
                 map=plpData.mapped$map)
  return(result)
}

# BuildCIReNN<-function(outcomes=ff::as.ffdf(population[,c('rowId','y')]),
#     covariates = result$data,
#     indexFolder=indexFolder){
# 
# }

#' Create setting for CIReNN model
#'
#' @param units         The number of units of RNN layer
#' @param indexFolder The directory where the results and intermediate steps are output
#' @param minCovariateFraction NULL or fraction. Minimum fraction of the population that shoul have a non-zero value for a covariate for that covariate to be kept. Set to 0 to don't filter on frequency
#'
#' @examples
#' \dontrun{
#' model.CIReNN <- setCIReNN()
#' }
#' @export
setCIReNN <- function(units=c(128, 64), recurrent_dropout=c(0.2), layer_dropout=c(0.2),
    lr =c(1e-4), decay=c(1e-5), outcome_weight = c(1.0), batch_size = c(100), 
    epochs= c(100), 
    #minCovariateFraction=0.01,
    indexFolder=file.path(getwd(),'CIReNN'), seed=NULL  ){
    
    # if(class(indexFolder)!='character')
    #     stop('IndexFolder must be a character')
    # if(length(indexFolder)>1)
    #     stop('IndexFolder must be one')
    # 
    # if(class(units)!='numeric')
    #     stop('units must be a numeric value >0 ')
    # if(units<1)
    #     stop('units must be a numeric value >0 ')
    # 
    # #if(length(units)>1)
    # #    stop('units can only be a single value')
    # 
    # if(class(recurrent_dropout)!='numeric')
    #     stop('dropout must be a numeric value >=0 and <1')
    # if( (recurrent_dropout<0) | (recurrent_dropout>=1))
    #     stop('dropout must be a numeric value >=0 and <1')
    # if(class(layer_dropout)!='numeric')
    #     stop('layer_dropout must be a numeric value >=0 and <1')
    # if( (layer_dropout<0) | (layer_dropout>=1))
    #     stop('layer_dropout must be a numeric value >=0 and <1')
    # if(class(lr)!='numeric')
    #     stop('lr must be a numeric value >0')
    # if(lr<=0)
    #     stop('lr must be a numeric value >0')
    # if(class(decay)!='numeric')
    #     stop('decay must be a numeric value >=0')
    # if(decay<=0)
    #     stop('decay must be a numeric value >=0')
    # if(class(outcome_weight)!='numeric')
    #     stop('outcome_weight must be a numeric value >=0')
    # if(outcome_weight<=0)
    #     stop('outcome_weight must be a numeric value >=0')
    # if(class(batch_size)!='numeric')
    #     stop('batch_size must be an integer')
    # if(batch_size%%1!=0)
    #     stop('batch_size must be an integer')
    # if(class(epochs)!='numeric')
    #     stop('epochs must be an integer')
    # if(epochs%%1!=0)
    #     stop('epochs must be an integer')
    # if(!class(seed)%in%c('numeric','NULL'))
    #     stop('Invalid seed')
    #if(class(UsetidyCovariateData)!='logical')
    #    stop('UsetidyCovariateData must be an TRUE or FALSE')

    result <- list(model='fitCIReNN', param=split(expand.grid(
      units=units, recurrent_dropout=recurrent_dropout, 
        layer_dropout=layer_dropout,
        lr =lr, decay=decay, outcome_weight=outcome_weight,epochs= epochs,
        #minCovariateFraction= minCovariateFraction,
        seed=ifelse(is.null(seed),'NULL', seed)),
    1:(length(units)*length(recurrent_dropout)*length(layer_dropout)*length(lr)*length(decay)*length(outcome_weight)*length(epochs)*max(1,length(seed)))),
      indexFolder=indexFolder,
      name='CIReNN'
      )


    #   list(units=units, recurrent_dropout=recurrent_dropout, 
    #     layer_dropout=layer_dropout,lr =lr, decay=decay, epochs= epochs, indexFolder=indexFolder),
    #     name='CIReNN'
    # )
    class(result) <- 'modelSettings' 
    
    return(result)
}


fitCIReNN <- function(plpData,population, param, search='grid', quiet=F,
                      outcomeId, cohortId, ...){
    # check plpData is coo format:
    if(!'ffdf'%in%class(plpData$covariates) || class(plpData)=='plpData.libsvm')
        stop('CIReNN requires plpData in coo format')
    
    metaData <- attr(population, 'metaData')
    if(!is.null(population$indexes))
        population <- population[population$indexes>0,]
    attr(population, 'metaData') <- metaData
    
    start<-Sys.time()

    # do cross validation to find hyperParameter
    
    hyperParamSel <- lapply(param, function(x) do.call(trainCIReNN, c(x, train=TRUE)  ))
    hyperSummary <- cbind(do.call(rbind, lapply(hyperParamSel, function(x) x$hyperSum)))
    hyperSummary <- as.data.frame(hyperSummary)
    hyperSummary$auc <- unlist(lapply(hyperParamSel, function (x) x$auc))
    hyperParamSel<-unlist(lapply(hyperParamSel, function(x) x$auc))
    
    #now train the final model and return coef
    bestInd <- which.max(abs(unlist(hyperParamSel)-0.5))[1]
    finalModel<-do.call(trainCIReNN, c(param[[bestInd]], train=FALSE))

    covariateRef <- ff::as.ram(plpData$covariateRef)
    incs <- rep(1, nrow(covariateRef)) 
    covariateRef$included <- incs
  
    #modelTrained <- file.path(outLoc) 
    param.best <- param[[bestInd]]
  
    comp <- start-Sys.time()

    # return model location 
    result <- list(model = finalModel,
                 trainCVAuc = -1, # ToDo decide on how to deal with this
                 hyperParamSearch = hyperSummary,
                 modelSettings = list(model='fitCIReNN',modelParameters=param.best),
                 metaData = plpData$metaData,
                 populationSettings = attr(population, 'metaData'),
                 outcomeId=population$outcomeId,
                 cohortId=population$cohortId,
                 varImp = covariateRef, 
                 trainingTime =comp,
                 dense=1
    )
    class(result) <- 'plpModel'
    attr(result, 'type') <- 'plp'
    attr(result, 'predictionType') <- 'binary'
  
    return(result)
}

trainCIReNN<-function(units=128, recurrent_dropout=0.2, layer_dropout=0.2,
    lr =1e-4, decay=1e-5, outcome_weight = 1.0, batch_size = 100, 
    epochs= 100, minCovariateFraction = 0.01,
    indexFolder=file.path(getwd(),'CIReNN'), seed=NULL, train=TRUE){
    
    #set seed
    if(!is.null(seed)){keras::use_session_with_seed(seed)}
    #check plpData is coo format:
    if(!'ffdf'%in%class(plpData$covariates) || class(plpData)=='plpData.libsvm')
        stop('CIReNN requires plpData in coo format')
    
    metaData <- attr(population, 'metaData')
    if(!is.null(population$indexes))
        population <- population[population$indexes>0,]
    attr(population, 'metaData') <- metaData
    
    start <- Sys.time()
    subanalysisId <- gsub(':','',gsub('-','',gsub(' ','',start)))

    #normalizing if UsetidyCovariateData is TRUE
    
    # if (UsetidyCovariateData) {covariates<-FeatureExtraction::tidyCovariateData(covariates=plpData$covariates, 
    #                                                            covariateRef=plpData$covariateRef,
    #                                                            populationSize = nrow(plpData$cohorts),
    #                                                            minFraction = 0.001,
    #                                                            normalize = TRUE,
    #                                                            removeRedundancy = TRUE)} 
    
    #covert data into sparse Array
    result<-toSparseArray(plpData,population,map=NULL)
    population$y <-keras::to_categorical(population$outcomeCount, length(unique(population$outcomeCount)))
    
    #if(!quiet)
    comp <- Sys.time() - start
    writeLines(paste0('Model CIReNN preprocessing - took:',  format(comp, digits=3)))
    
    ##SET the Model
    ##single-layer gru
    model <- keras::keras_model_sequential()
    model %>%
      keras::layer_gru(units=units, recurrent_dropout = recurrent_dropout,
                       input_shape = c(dim(result$data)[2],dim(result$data)[3]), #input_shape = c(dim(data)[2],dim(data)[3]), #time step x number of features
                       return_sequences=FALSE#,stateful=TRUE
      ) %>%
      keras::layer_dropout(layer_dropout) %>%
      keras::layer_dense(units=2, activation='softmax')
    
    model %>% keras::compile(
      loss = 'binary_crossentropy',
      metrics = c('accuracy'),
      optimizer = keras::optimizer_rmsprop(lr = lr,decay = decay)
    )
    earlyStopping=keras::callback_early_stopping(monitor = "val_loss", patience=max(10,round(epochs/50)),mode="auto",min_delta = 0)
    reduceLr=keras::callback_reduce_lr_on_plateau(monitor="val_loss", factor =0.1, 
                                                  patience = max(5,round(epochs/100)),mode = "auto", epsilon = 1e-5, cooldown = 0, min_lr = 0)
    class_weight=list("0"=1,"1"=outcome_weight)
    
    ##Train the model 
    ##If data_augmentation is TRUE, batch-size data is loaded on the R momery from sparse array by sampling generator
    data_augmentation=TRUE
    data_sampling_randome=TRUE
    validation_split=0.2
    if(data_augmentation){
      #one-hot encoding
      data <-result$data[population$rowId,,]
      
      #Extract validation set first
      val_rows<-sample(1:length(population$rowId), length(population$rowId)*validation_split, replace=FALSE)
      val_data=list(x_val=as.array(data[val_rows,,]), y_val = population$y[val_rows,])
      
      sampling_generator<-function(data,population, batch_size, val_rows){
        function(){
          gc()
          targetId<-population$rowId[-population$rowId[val_rows]]
          rows<-sample(1: (length(targetId)), batch_size, replace=FALSE)
          list(as.array(data[rows,,]), population$y[rows,])
        }
      }
      
      history<-model %>% keras::fit_generator(sampling_generator(data,population,batch_size,val_rows),
                                              steps_per_epoch = length(population$rowId)/batch_size,
                                              epochs=epochs,
                                              validation_data=list(as.array(data[val_rows,,]), population$y[val_rows,]),
                                              #validation_data = val_data,
                                              #validation_split=0.2,
                                              callbacks=list(earlyStopping,reduceLr),
                                              class_weight=class_weight
                                              #,validation_data = list(x_val=validation.set,y_val = validation.y)
                                              #,shuffle=TRUE
      )
      }else{
      data <-result$data[population$rowId,,]
      data<-as.array(data)
      history<-model %>% keras::fit(data, population$y , epochs=epochs,
                                    batch_size =batch_size
                                    ,validation_split=0.2
                                    #,validation_data = list(x_val=validation.set,y_val = validation.y)
                                    #,shuffle=TRUE
                                    ,callbacks=list(earlyStopping,reduceLr)
                                    ,class_weight=class_weight
      )
    }

    comp <- Sys.time() - start
    writeLines(paste0('Model CIReNN train - took:',  format(comp, digits=3)))
    
    #keras::save_model_hdf5(model, filepath, overwrite = TRUE,include_optimizer = TRUE)
    
    #eval<- model %>% keras::evaluate(data, population$y) 
    #pred_class<- model %>% keras::predict_classes(data)
    
    
    #value<- stats::predict(model,data[population$indexes==index,,])  need to be revised, how can I differentiate train and test?
    #pROC::roc(prediction$y[,2],prediction$value[,2])

    
    pred <- stats::predict(model, data)
    prediction <- population
    prediction$value <- pred[,2]
    prediction$y<-prediction$y[,2]
    attr(prediction, "metaData") <- list(predictionType = "binary") 
    auc <- computeAuc(prediction)
    
    #value<- stats::predict(model,data)
    #pred<-data.frame(rowId=population$rowId,
    #                 outcomeCount=population$outcomeCount,
                     #indexes = indexes,
    #                 value = value[,2]) #If the outcome is not binary, the value should be changed (just using value)

    #prediction <- population
    #prediction$value <- pred
    attr(prediction, "metaData") <- list(predictionType = "binary") 
    hyperPerm <- auc

    varImp<- ff::as.ram(plpData$covariateRef)
    varImp$covariateValue <- rep(0, nrow(varImp))
    
    param.val <- paste0('units: ',units,' --recurrent_dropout: ', recurrent_dropout,
                        ' --layer_dropout: ',layer_dropout,'-- lr: ', lr,
                        ' --decay: ', decay, ' --batch_size: ',batch_size, ' --planned epochs: ', epochs, ' --outcome_weight:', outcome_weight)
    writeLines('==========================================')
    writeLines(paste0('CIReNN with parameters:', param.val,' obtained an AUC of ',auc))
    writeLines('==========================================')
    
    result <- list(model=model,
                   auc=auc,
                   hyperSum = unlist(list(units=units, recurrent_dropout=recurrent_dropout, 
                                          layer_dropout=layer_dropout,lr =lr, decay=decay,
                                          batch_size = batch_size, outcome_weight=outcome_weight,
                                          epochs= epochs)),
                   metaData = plpData$metaData,
                   populationSettings = attr(population, 'metaData'),
                   outcomeId=population$outcomeId,
                   cohortId=population$cohortId,
                   varImp = varImp,
                   trainingTime =comp,
                   history=history$metrics
    )
    # hyperSumHistory<-cbind(as.data.frame(history$metrics),as.data.frame(unlist(list(units=units, recurrent_dropout=recurrent_dropout, 
    #                                                          layer_dropout=layer_dropout,lr =lr, decay=decay,
    #                                                          batch_size = batch_size, outcome_weight=outcome_weight,
    #                                                          epochs= epochs))))
    
    modelPath<-file.path(analysisPath,'savedModel',subanalysisId)
    if(!dir.exists(modelPath)){dir.create(modelPath,recursive=T)}
    saveRDS(result,file=file.path(modelPath,"CIReNN_history.rds"))
    write.csv(as.data.frame(history$metrics),file=file.path(modelPath,"CIReNN_history.csv"))
    
    class(result) <- 'plpModel'
    attr(result, 'type') <- 'CIReNN'
    attr(result, 'predictionType') <- 'binary'
    
    gc()
    
    return(result)
}