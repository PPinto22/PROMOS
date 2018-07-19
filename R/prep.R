# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(jsonlite)
library(data.table)
source('util.R')

# Config
INPUT <- '../data/2weeks/ColetasPROMOSBESTTreated2018-06-13.json'
OUT_DIR <- add_trailing_slash('../data/2weeks')
OUT_ID <- 'best'
TARGET <- "target"
POSITIVE_CLASS <- "Sale"
DATE <- "date_added_utc"

# Read data
# If data is separated by rows
json_txt <- readLines(INPUT)
json_formatted <- paste('[', paste(json_txt, collapse=','), ']', sep='')
dt <- as.data.table(jsonlite::fromJSON(json_formatted))
# If data is valid JSON
# dt <- as.data.table(jsonlite::fromJSON(INPUT))

dt$id <- NULL

# Replace empty cells with the keyword EMPTY
dt <- dt[, lapply(.SD, function(x) replace(x, which(x==''), 'empty'))]

# Transform TARGET from categorical to numeric
dt[[TARGET]] <- replace(dt[[TARGET]], dt[[TARGET]]!=POSITIVE_CLASS, 0)
dt[[TARGET]] <- replace(dt[[TARGET]], dt[[TARGET]]==POSITIVE_CLASS, 1)
dt[[TARGET]] <- as.numeric(dt[[TARGET]])

# Rename columns
names(dt)[names(dt) == 'country_name'] <- 'country'
names(dt)[names(dt) == 'date_added_utc'] <- 'timestamp'
DATE <- "timestamp"

# Save csv
write.table(dt, file=paste(OUT_DIR, OUT_ID, '.csv', sep=''), sep=',', row.names = FALSE)
