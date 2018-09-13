# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
library(pROC)

TRAIN_FILES <- c('../data/2weeks/best_idf_train.csv')
TEST_FILES <- c('../data/2weeks/best_idf_test.csv')
ENCODINGS <- c('idf')
MODES <- c('best')
options(digits=5)

read_data <- function(path){
  dt = data.table(read.csv(path, header=TRUE, sep=','))
  dt$timestamp <- NULL
  return(dt)
}

summary_df <- data.frame(algorithm=character(), mode=character(), encoding=character(), auc=double(), time=double())

train_dts <- lapply(TRAIN_FILES, read_data)
test_dts <- lapply(TEST_FILES, read_data)

# linear regression
for (i in 1:length(TRAIN_FILES)) {
  train_dt = train_dts[[i]]
  test_dt = test_dts[[i]]
  mode = MODES[[i]]
  encoding = ENCODINGS[[i]]
  
  time <- system.time(glm_model <- glm(target ~ ., train_dt,  family = "binomial"))[[3]]
  predictions <- predict(glm_model, test_dt)
  auc_score <- auc(roc(test_dt$target, predictions))
  summary_row <- list(algorithm='lr', mode=mode, encoding=encoding, auc=auc_score, time=time)
  summary_df <- rbind(summary_df, summary_row)
}

# xgboost
# mlp
# svm
# random forest
