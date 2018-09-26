# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
library(rminer)
library(pROC)
library(optparse)

option_list = list(make_option(c("-o", "--out"), type="character", default="out/models.csv", help="output file name [default: %default]", metavar="FILE")); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

read_data <- function(path, to_factor=F){
  dt = data.table(read.csv(path, header=TRUE, sep=','))
  dt$timestamp <- NULL
  if(to_factor) dt$target <- as.factor(dt$target)
  return(dt)
}

set_paths <- function(instances){
  ret = lapply(instances, function(instance){
    windows = instance[[5]]
    if(windows <= 1){
      instance[[1]] <- list(paste(instance[[1]], '.csv', sep=''))
      instance[[2]] <- list(paste(instance[[2]], '.csv', sep=''))
    }
    else{
      instance[[1]] <- as.list(CJ(instance[[1]], 0:(windows-1), sorted = FALSE)[, paste(V1, '(', V2, ')', '.csv', sep ="")])
      instance[[2]] <- as.list(CJ(instance[[2]], 0:(windows-1), sorted = FALSE)[, paste(V1, '(', V2, ')', '.csv', sep ="")])
    }
    return(instance)
  })
  return(ret)
}

# Train; test; encoding; mode; windows
OUT <- opt$out
INSTANCES <- set_paths(list(
  # list('../data/2weeks/sw/best_idf_train', '../data/2weeks/sw/best_idf_test', 'idf', 'best', 10),
  # list('../data/2weeks/sw/best_raw_train', '../data/2weeks/sw/best_raw_test', 'raw', 'best', 10),
  list('../data/2weeks/sw/best_pcp_train', '../data/2weeks/sw/best_pcp_test', 'pcp', 'best', 10)
  
  # list('../data/2weeks/best_idf_train', '../data/2weeks/best_idf_test', 'idf', 'best'),
  # list('../data/2weeks/best_raw_train', '../data/2weeks/best_raw_test', 'raw', 'best'),
  # list('../data/2weeks/best_pcp_train', '../data/2weeks/best_pcp_test', 'pcp', 'best'),
  # list('../data/2weeks/test_idf_train', '../data/2weeks/test_idf_test', 'idf', 'test'),
  # list('../data/2weeks/test_raw_train', '../data/2weeks/test_raw_test', 'raw', 'test'),
  # list('../data/2weeks/test_pcp_train', '../data/2weeks/test_pcp_test', 'pcp', 'test')
))
# str(INSTANCES)
# ALGORITHMS <- c('lr', 'naivebayes', 'mlp', 'xgboost')
ALGORITHMS <- list('mlp')
options(digits=5)

summary_df <- rbindlist(lapply(ALGORITHMS, function(alg){
  rbindlist(lapply(1:length(INSTANCES), function(i){
    train_files = INSTANCES[[i]][[1]]
    test_files = INSTANCES[[i]][[2]]
    encoding = INSTANCES[[i]][[3]]
    mode = INSTANCES[[i]][[4]]
    
    rbindlist(lapply(1:length(train_files), function(j){
      print(paste('[ALGORITHM]', alg, '[MODE]', mode, '[ENCODING]', encoding, '[WINDOW]', j, '...'))
      
      train_dt = read_data(train_files[[j]])
      test_dt = read_data(test_files[[j]])

      summary_row = tryCatch({
        time <- system.time(model <- fit(target ~ ., train_dt,  model = alg, task = 'prob', MaxNWts=5000))[[3]] # MaxNWts is for MLP
        predictions <- predict(model, test_dt)
        if(alg != 'mlp') { predictions <- predictions[,2] }
        predictions <- as.vector(predictions)
        auc_score <- as.double(auc(roc(test_dt$target, predictions)))
        summary_row <- data.frame(algorithm=alg, mode=mode, encoding=encoding, window=j, auc=auc_score, time=time/60, error=' ')
        return(summary_row)
      }, error = function(e){
        print(e)
        summary_row <- data.frame(algorithm=alg, mode=mode, encoding=encoding, window=j, auc=NA, time=NA, error=as.character(e))
        return(summary_row)
      })
      return(summary_row)
    })) # /windows
  })) # /instances
})) # /algorithms

dir.create(file.path(dirname(OUT)), recursive=TRUE, showWarnings=FALSE)
write.table(summary_df, file=OUT, sep=',', row.names = FALSE)