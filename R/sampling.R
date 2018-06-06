# install.packages("rstudioapi")
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

data <- read.csv("../data/old/data.csv", header = TRUE, sep = ";")

# data.train.index = sample(1:nrow(data),size=ceiling(0.7*nrow(data))) 
# data.train <- data[data.train.index,] 
# data.test <- data[-data.train.index,]
# 
# write.table(data.train, file="../data/data_train.csv", row.names = FALSE, sep = ";")
# write.table(data.test , file="../data/data_test.csv",  row.names = FALSE, sep = ";")
# 
# 
# data1 <- read.csv("../data/data_train.csv", header = TRUE, sep = ";")
# data2 <- read.csv("../data/data_test.csv", header = TRUE, sep = ";")

PROBS <- c(0.6, 0.2, 0.2)
OUT_DIR <- "../data/old/"
OUT_FILES <- paste( OUT_DIR, c("train", "test1", "test2"), ".csv", sep='' )


samples_idx = sample(length(PROBS), nrow(data), replace=TRUE, prob=PROBS)
samples = setNames(split(data, samples_idx), OUT_FILES)

table(samples_idx)

for(sname in names(samples)){
  print(sname)
  
}