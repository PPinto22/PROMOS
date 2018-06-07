# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(jsonlite)
library(data.table)
library(hash)

TARGET <- "target"
POSITIVE_CLASS <- "Sale"
DATE <- "date_added_utc"
INPUT <- '../data/week1_best/ColetasPROMOSBESTTreated2018-05-16.json'
OUTPUT <- '../data/week1_best/week1_best.csv'

# Read data
dt <- as.data.table(fromJSON(INPUT))
dt$id <- NULL

# Rename country_name to country
names(dt)[names(dt) == 'country_name'] <- 'country'

# Replace empty cells with the keyword EMPTY
dt <- dt[, lapply(.SD, function(x) replace(x, which(x==''), 'empty'))]

# Transform TARGET from categorical to numeric
dt[[TARGET]] <- replace(dt[[TARGET]], dt[[TARGET]]==POSITIVE_CLASS, 1)
dt[[TARGET]] <- replace(dt[[TARGET]], dt[[TARGET]]!=POSITIVE_CLASS, 0)
dt[[TARGET]] <- as.numeric(dt[[TARGET]])

idfDeep=function(v)
{
  tf=table(v)
  N=length(v)
  idf=log(N/tf)
  idf_hash=hash(keys=names(idf), values=idf)
  
  return(sapply(v, function(x){idf_hash[[as.character(x)]]}))
}
# v <- c('a','b','b','c','c','c','d','d','d','d')
# idfDeep(v)

cols <- colnames(dt)[!colnames(dt) %in% c(TARGET, DATE)]
dt[, (cols) := lapply(.SD, idfDeep), .SDcols = cols]

write.table(dt, file = OUTPUT, sep=',', row.names = FALSE)
