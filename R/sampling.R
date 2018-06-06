# install.packages("rstudioapi")
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

PROBS <- c(0.6, 0.2, 0.2)
OUT_DIR <- "../data/old/"
OUT_FILES <- paste( OUT_DIR, c("train", "test1", "test2"), ".csv", sep='' )

data <- read.csv("../data/old/data.csv", header = TRUE, sep = ";")

samples_idx = sample(length(PROBS), nrow(data), replace=TRUE, prob=PROBS)
samples = setNames(split(data, samples_idx), OUT_FILES)

for(sname in names(samples)){
  write.table(samples[[sname]], file=sname, sep=',', row.names = FALSE)
}