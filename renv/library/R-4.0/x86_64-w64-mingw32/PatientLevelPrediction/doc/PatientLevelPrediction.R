## ---- echo = TRUE, message = FALSE, warning = FALSE,tidy=FALSE,eval=FALSE-----
## install.packages("drat")
## drat::addRepo("OHDSI")
## install.packages("PatientLevelPrediction")


## ---- echo = TRUE, message = FALSE, warning = FALSE,tidy=FALSE,eval=FALSE-----
## install.packages('devtools')
## devtools::install_github("OHDSI/FeatureExtraction")
## devtools::install_github('ohdsi/PatientLevelPrediction')


## ---- echo = TRUE, message = FALSE, warning = FALSE,tidy=FALSE,eval=FALSE-----
## library(PatientLevelPrediction)
## reticulate::install_miniconda()
## configurePython(envname='r-reticulate', envtype='conda')
## 


## ---- echo = TRUE, message = FALSE, warning = FALSE,tidy=FALSE,eval=FALSE-----
## devtools::install_github("rstudio/keras")
## library(keras)
## install_keras()


## ---- echo = TRUE, message = FALSE, warning = FALSE,tidy=FALSE,eval=FALSE-----
## reticulate::conda_install(envname='r-reticulate', packages = c('scikit-survival'), forge = TRUE, pip = FALSE, pip_ignore_installed = TRUE, conda = "auto", channel = 'sebp')
## 


## ---- echo = TRUE, message = FALSE, warning = FALSE,tidy=FALSE,eval=FALSE-----
## reticulate::conda_install(envname='r-reticulate', packages = c('pytorch', 'torchvision', 'cpuonly'), forge = TRUE, pip = FALSE, channel = 'pytorch', pip_ignore_installed = TRUE, conda = 'auto')
## 


## ----eval=FALSE---------------------------------------------------------------
## library(PatientLevelPrediction)
## library(DatabaseConnector)
## connectionDetails <- createConnectionDetails(dbms = 'sql_server',
##                                              user = 'username',
##                                              password = 'hidden',
##                                              server = 'your server',
##                                              port = 'your port')
## PatientLevelPrediction::checkPlpInstallation(connectionDetails = connectionDetails,
##                                              python = T)


## ----eval=FALSE---------------------------------------------------------------
## library(PatientLevelPrediction)
## library(DatabaseConnector)
## connectionDetails <- createConnectionDetails(dbms = 'sql_server',
##                                            user = 'username',
##                                            password = 'hidden',
##                                            server = 'your server',
##                                            port = 'your port')
## PatientLevelPrediction::checkPlpInstallation(connectionDetails = connectionDetails,
##                                              python = F)


## ----eval=FALSE---------------------------------------------------------------
## usethis::edit_r_profile()


## ----eval=FALSE---------------------------------------------------------------
## Sys.setenv(PATH = paste("your python bin location", Sys.getenv("PATH"), sep=":"))


## ----eval=FALSE---------------------------------------------------------------
## reticulate::conda_list()


## ----eval=FALSE---------------------------------------------------------------
## system("python --version")


## ----eval=FALSE---------------------------------------------------------------
##   usethis::edit_r_profile()


## ----eval=FALSE---------------------------------------------------------------
## Sys.setenv(PATH = paste("your python bin location", Sys.getenv("PATH"), sep=":"))


## ----eval=FALSE---------------------------------------------------------------
## system("python --version")

