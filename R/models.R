# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
library(rminer)
library(pROC)
library(optparse)

option_list = list(make_option(c("-o", "--out"), type="character", default="out/models.csv", help="output file name [default: %default]", metavar="FILE")); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

read_data <- function(path){
  dt = data.table(read.csv(path, header=TRUE, sep=','))
  dt$timestamp <- NULL
  dt$target <- as.factor(dt$target)
  return(dt)
}

# Train; test; encoding; mode
OUT <- opt$out
INSTANCES <- list(
  # c('../data/2weeks/best_idf_mini.csv', '../data/2weeks/best_idf_mini.csv', 'idf', 'best')

  c('../data/2weeks/best_idf_train.csv', '../data/2weeks/best_idf_test.csv', 'idf', 'best'),
  c('../data/2weeks/best_raw_train.csv', '../data/2weeks/best_raw_test.csv', 'raw', 'best'),
  c('../data/2weeks/best_pcp_train.csv', '../data/2weeks/best_pcp_test.csv', 'pcp', 'best'),
  c('../data/2weeks/test_idf_train.csv', '../data/2weeks/test_idf_test.csv', 'idf', 'test'),
  c('../data/2weeks/test_raw_train.csv', '../data/2weeks/test_raw_test.csv', 'raw', 'test'),
  c('../data/2weeks/test_pcp_train.csv', '../data/2weeks/test_pcp_test.csv', 'pcp', 'test')
)
ALGORITHMS <- c('lr', 'naivebayes', 'mlp', 'xgboost')
options(digits=5)

train_dts <- lapply(INSTANCES, function(x) read_data(x[1]))
test_dts <- lapply(INSTANCES, function(x) read_data(x[2]))

summary_df <- rbindlist(lapply(ALGORITHMS, function(alg){
  rbindlist(lapply(1:length(INSTANCES), function(i){
    train_dt = train_dts[[i]]
    test_dt = test_dts[[i]]
    mode = INSTANCES[[i]][3]
    encoding = INSTANCES[[i]][4]
    
    print(paste('[ALGORITHM]', alg, '[MODE]', mode, '[ENCODING]', encoding, '...'))
    
    summary_row = tryCatch({
      time <- system.time(model <- fit(target ~ ., train_dt,  model = alg, task = 'prob'))[[3]]
      predictions <- as.vector(predict(model, test_dt)[,2])
      auc_score <- auc(roc(test_dt$target, predictions))
      summary_row <- data.frame(algorithm=alg, mode=mode, encoding=encoding, auc=auc_score, time=time, error='')
      return(summary_row)
    }, error = function(e){
      summary_row <- data.frame(algorithm=alg, mode=mode, encoding=encoding, auc=NA, time=NA, error=as.character(e))
      return(summary_row)
    })
    return(summary_row)
  }))
}))

# # FIXME: DEBUG
# train_dt = train_dts[[1]]
# test_dt = test_dts[[1]]
# mode = INSTANCES[[1]][3]
# encoding = INSTANCES[[1]][4]
# time <- system.time(model <- fit(target ~ ., train_dt,  model = 'naivebayes', task = 'prob'))[[3]]
# predictions <- as.vector(predict(model, test_dt)[,2])
# auc(roc(test_dt$target, predictions))

dir.create(file.path(dirname(OUT)), recursive=TRUE, showWarnings=FALSE)
write.table(summary_df, file=OUT, sep=',', row.names = FALSE)
