

drat::addRepo("OHDSI")
install.packages("PatientLevelPrediction")
install.packages(c("AUC", "aws.s3", "BigKnn", "devtools", 
                   "diagram", "DT", "gnm", "keras", "markdown", 
                   "Metrics", "officer", "pkgdown", "plotly", 
                   "pROC", "ResourceSelection", "rmarkdown", 
                   "scoring", "shiny", "shinycssloaders",
                   "shinydashboard", "shinyWidgets", "survAUC", 
                   "survminer", "tensorflow", "testthat", "xgboost"))
remotes::install_github("OhdsiRTools")
remotes::install_github("ohdsi/OhdsiRTools")
renv::snapshot()
