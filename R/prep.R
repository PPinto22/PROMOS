# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(jsonlite)
library(data.table)
library(optparse)
source('util.R')

# Config
option_list = list(
  make_option(c("-f", "--file"), type="character", default='../data/2weeks/ColetasPROMOSTESTTreated2018-06-13.json', 
              help="dataset file name", metavar="FILE"),
  make_option(c("-o", "--out"), type="character", default="../data/2weeks/best.csv", 
              help="output file name [default= %default]", metavar="character")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

INPUT <- opt$file
OUT <- opt$out
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

# REPLACE empty cells with the keyword EMPTY
# dt <- dt[, lapply(.SD, function(x) replace(x, which(x==''), 'empty'))]
# DELETE empty cells
empty_rows <- apply(dt, 1, function(x) any(x==''))
dt <- dt[!empty_rows, ]

# Transform TARGET from categorical to numeric
dt[[TARGET]] <- replace(dt[[TARGET]], dt[[TARGET]]!=POSITIVE_CLASS, 0)
dt[[TARGET]] <- replace(dt[[TARGET]], dt[[TARGET]]==POSITIVE_CLASS, 1)
dt[[TARGET]] <- as.numeric(dt[[TARGET]])

# Rename columns
names(dt)[names(dt) == 'country_name'] <- 'country'
names(dt)[names(dt) == 'date_added_utc'] <- 'timestamp'
DATE <- "timestamp"

# Save csv
dir.create(file.path(dirname(OUT)), recursive=TRUE, showWarnings=FALSE)
write.table(dt, file=OUT, sep=',', row.names = FALSE)
