library(nnet)
library(data.table)
library(pROC)

read_data <- function(path, to_factor=F){
  dt = data.table(read.csv(path, header=TRUE, sep=','))
  dt$timestamp <- NULL
  if(to_factor) dt$target <- as.factor(dt$target)
  return(dt)
}

train_dt <- read_data('/home/pedro/Desktop/PROMOS/data/2weeks/best_idf_small_train.csv')
test_dt <- read_data('/home/pedro/Desktop/PROMOS/data/2weeks/best_idf_small_test.csv')

net = nnet(target ~ ., train_dt, size=11, maxit=200)

# Test
predictions <- predict(net, test_dt)
predictions <- as.vector(predictions)
auc_score <- as.double(auc(roc(test_dt$target, predictions)))

# Train
predictions <- predict(net, train_dt)
predictions <- as.vector(predictions)
auc_score_train <- as.double(auc(roc(train_dt$target, predictions)))

print(auc_score_train)
print(auc_score)
