# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
library(rminer)
library(pROC)

# Train; test; encoding; mode
INSTANCES <- list(
  c('../data/2weeks/best_idf_train.csv', '../data/2weeks/best_idf_test.csv', 'idf', 'best')
)
ALGORITHMS <- c('lr', 'randomForest', 'svm', 'mlp', 'xgboost')
options(digits=5)

read_data <- function(path){
  dt = data.table(read.csv(path, header=TRUE, sep=','))
  dt$timestamp <- NULL
  return(dt)
}

train_dts <- lapply(INSTANCES, function(x) read_data(x[1]))
test_dts <- lapply(INSTANCES, function(x) read_data(x[2]))

summary_df <- data.frame(algorithm=character(), mode=character(), encoding=character(), auc=double(), time=double())
  
for(alg in ALGORITHMS){
  for (i in 1:length(INSTANCES)){
    train_dt = train_dts[[i]]
    test_dt = test_dts[[i]]
    mode = INSTANCES[[i]][3]
    encoding = INSTANCES[[i]][4]
    
    time <- system.time(model <- fit(target ~ ., train_dt,  model = alg, task = 'prob'))[[3]]
    predictions <- predict(model, test_dt)
    auc_score <- auc(roc(test_dt$target, predictions))
    summary_row <- list(algorithm=alg, mode=mode, encoding=encoding, auc=auc_score, time=time)
    summary_df <- rbind(summary_df, summary_row)
  }
}
