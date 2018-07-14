# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

INPUT <- '../data/2weeks/best_pcp.csv'
OUT_DIR <- '../data/2weeks/'
OUT_ID <- 'best_pcp'
OUT_FILES <- paste( OUT_DIR, OUT_ID, c('_train', '_val', '_test'), '.csv', sep='' )
PROBS <- c(0.6, 0.2, 0.2)

data <- read.csv(INPUT, header = TRUE, sep = ',')

samples_idx = sample(length(PROBS), nrow(data), replace=TRUE, prob=PROBS)
samples = setNames(split(data, samples_idx), OUT_FILES)

for(sname in names(samples)){
  dt = samples[[sname]]
  write.table(dt, file=sname, sep=',', row.names = FALSE)
  print(nrow(dt[dt$target==0,]))
  print(nrow(dt[dt$target==1,]))
}