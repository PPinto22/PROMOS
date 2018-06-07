# install.packages("rstudioapi")
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

PROBS <- c(0.6, 0.2, 0.2)
INPUT <- '../data/week1_best/week1_best.csv'
OUT_DIR <- "../data/week1_best/"

OUT_FILES <- paste( OUT_DIR, c("train", "test1", "test2"), ".csv", sep='' )

data <- read.csv(INPUT, header = TRUE, sep = ",")

samples_idx = sample(length(PROBS), nrow(data), replace=TRUE, prob=PROBS)
samples = setNames(split(data, samples_idx), OUT_FILES)

for(sname in names(samples)){
  dt = samples[[sname]]
  write.table(dt, file=sname, sep=',', row.names = FALSE)
  # print(nrow(dt[dt$target==0,]))
  # print(nrow(dt[dt$target==1,]))
}