#install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

data <- read.csv("../data/data.csv", header = TRUE, sep = ";")

data.train.index = sample(1:nrow(data),size=ceiling(0.7*nrow(data))) 
data.train <- data[data.train.index,] 
data.test <- data[-data.train.index,]

write.table(data.train, file="../data/data_train.csv", row.names = FALSE, sep = ";")
write.table(data.test , file="../data/data_test.csv",  row.names = FALSE, sep = ";")


data1 <- read.csv("../data/data_train.csv", header = TRUE, sep = ";")
data2 <- read.csv("../data/data_test.csv", header = TRUE, sep = ";")
