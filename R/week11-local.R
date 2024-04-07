#Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(haven)
library(tidyverse)
library(caret)
library(parallel) #added for parallelization
library(doParallel) #added for parallelization
library(tictoc) #to track times
set.seed(123123)

#Data Import and Cleaning
gss_import_tbl <- read_spss("../data/GSS2016.sav") %>%
  filter(!is.na(MOSTHRS)) %>%
  rename("work hours" = MOSTHRS) %>% #
  select(-HRS1, -HRS2)

gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl)) < .75 * nrow(gss_import_tbl)] %>%
  mutate(across(everything(), as.numeric)) #update the code to reflect demonstration since grades havent been released and i am not sure if i did everything correct

#Visualization
ggplot(gss_tbl, aes(x = `work hours`)) + 
  geom_histogram()
  labs(title = "Distribution of Work Hours", x = "Work Hours", y = "Frequency") 

# Analysis
##updated all code to match class demo since i know it is correct 
holdout_indices <- createDataPartition(gss_tbl$`work hours`,
                                       p = .25,
                                       list = T)$Resample1
test_tbl <- gss_tbl[holdout_indices,]
training_tbl <- gss_tbl[-holdout_indices,]

training_folds <- createFolds(training_tbl$`work hours`)

tic()
ols_model1 <- train(
  `work hours` ~ .,
  training_tbl,
  method="lm",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
olstoc <- toc()
ols_time <- olstoc$toc - olstoc$tic
ols_model1
cv_m1 <- ols_model1$results$Rsquared
holdout_m1 <- cor(
  predict(ols_model1, test_tbl, na.action = na.pass),
  test_tbl$`work hours`
)^2

tic()
en_model2 <- train(
  `work hours` ~ .,
  training_tbl,
  method="glmnet",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
en_toc <- toc()
en_time <- en_toc$toc - en_toc$tic
en_model2
cv_m2 <- max(en_model2$results$Rsquared)
holdout_m2 <- cor(
  predict(en_model2, test_tbl, na.action = na.pass),
  test_tbl$`work hours`
)^2

tic()
rf_model3 <- train(
  `work hours` ~ .,
  training_tbl,
  method="ranger",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
rf_toc <- toc()
rf_time <- rf_toc$toc - rf_toc$tic
rf_model3
cv_m3 <- max(rf_model3$results$Rsquared)
holdout_m3 <- cor(
  predict(rf_model3, test_tbl, na.action = na.pass),
  test_tbl$`work hours`
)^2

tic()
xgb_model4 <- train(
  `work hours` ~ .,
  training_tbl,
  method="xgbLinear",
  na.action = na.pass,
  tuneLength = 1,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
xgb_toc <- toc()
xgb_time <- xgb_toc$toc - xgb_toc$tic
xgb_model4
cv_m4 <- max(xgb_model4$results$Rsquared)
holdout_m4 <- cor(
  predict(xgb_model4, test_tbl, na.action = na.pass),
  test_tbl$`work hours`
)^2

summary(resamples(list(ols_model1, en_model2, rf_model3, xgb_model4)), metric="Rsquared")
dotplot(resamples(list(ols_model1, en_model2, rf_model3, xgb_model4)), metric="Rsquared")

## Parallelization
local_cluster <- makeCluster(detectCores() - 1) # based on lecture 
registerDoParallel(local_cluster)

tic()
ols_model1.1 <- train(
  `work hours` ~ .,
  training_tbl,
  method="lm",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
olstoc_p <- toc()
ols_time_p <- olstoc_p$toc - olstoc_p$tic
ols_model1.1
cv_m1_p <- ols_model1.1$results$Rsquared
holdout_m1_p <- cor(
  predict(ols_model1.1, test_tbl, na.action = na.pass),
  test_tbl$`work hours`
)^2

tic()
en_model2.2 <- train(
  `work hours` ~ .,
  training_tbl,
  method="glmnet",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
en_toc_p <- toc()
en_time_p <- en_toc_p$toc - en_toc_p$tic
en_model2.2
cv_m2_p <- max(en_model2.2$results$Rsquared)
holdout_m2_p <- cor(
  predict(en_model2.2, test_tbl, na.action = na.pass),
  test_tbl$`work hours`
)^2

tic()
rf_model3.3 <- train(
  `work hours` ~ .,
  training_tbl,
  method="ranger",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
rf_toc_p <- toc()
rf_time_p <- rf_toc_p$toc - rf_toc_p$tic
rf_model3.3
cv_m3_p <- max(rf_model3.3$results$Rsquared)
holdout_m3_p <- cor(
  predict(rf_model3.3, test_tbl, na.action = na.pass),
  test_tbl$`work hours`
)^2

tic()
xgb_model4.4 <- train(
  `work hours` ~ .,
  training_tbl,
  method="xgbLinear",
  na.action = na.pass,
  tuneLength = 1,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
xgb_toc_p <- toc()
xgb_time_p <- xgb_toc_p$toc - xgb_toc_p$tic
xgb_model4.4
cv_m4_p <- max(xgb_model4.4$results$Rsquared)
holdout_m4_p <- cor(
  predict(xgb_model4.4, test_tbl, na.action = na.pass),
  test_tbl$`work hours`
)^2

summary(resamples(list(ols_model1.1, en_model2.2, rf_model3.3, xgb_model4.4)), metric="Rsquared")
dotplot(resamples(list(ols_model1.1, en_model2.2, rf_model3.3, xgb_model4.4)), metric="Rsquared")

stopCluster(local_cluster) #stop cluster
registerDoSEQ() #register a sequential parallel backend 

# Publication
make_it_pretty <- function (formatme) {
  formatme <- formatC(formatme, format="f", digits=2)
  formatme <- str_remove(formatme, "^0")
  return(formatme)
}

table1_tbl <- tibble(
  algo = c("regression","elastic net","random forests","xgboost"),
  cv_rqs = c(
    make_it_pretty(cv_m1),
    make_it_pretty(cv_m2),
    make_it_pretty(cv_m3),
    make_it_pretty(cv_m4)
  ),
  ho_rqs = c(
    make_it_pretty(holdout_m1),
    make_it_pretty(holdout_m2),
    make_it_pretty(holdout_m3),
    make_it_pretty(holdout_m4)
  )
)

table2_tbl <- tibble(
  algo = c("OLS regression", "Elastic Net", "Random Forest", "eXtreme Gradient Boosting"),
  original = c(ols_time, en_time, rf_time, xgb_time),
  parallelized = c(ols_time_p, en_time_p, rf_time_p, xgb_time_p),
)

print(table2_tbl)
#Answers

## 1:
## 2:
## 3: