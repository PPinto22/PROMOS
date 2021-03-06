# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(jsonlite)
library(data.table)
library(hash)
library(dummies)
source('util.R')

# Inputs
INPUT <- '../data/2weeks/best.csv'
TARGET <- "target"
DATE <- "timestamp"

# Outputs
OUT_DIR <- add_trailing_slash('../data/2weeks')
OUT_ID <- append_str('best_temp', '_')
OUT_IDF <- FALSE
OUT_PCP <- TRUE

# Read data
dt <- data.table(read.csv(file=INPUT, header=TRUE, sep=','))

# Input columns
cols <- colnames(dt)[!colnames(dt) %in% c(TARGET, DATE)]

# To factor
dt[, (cols) := lapply(.SD, as.factor), .SDcols = cols]

idfDeep <- function(v)
{
  tf=table(v)
  N=length(v)
  idf=log(N/tf)
  idf_hash=hash(keys=names(idf), values=idf)
  
  return(sapply(v, function(x){idf_hash[[as.character(x)]]}))
}

PCP <- function(f, percentage = 0.05)
{
  tf=table(f)
  tf=sort(tf,decreasing=T)
  CPercent = ceiling(length(f) * percentage)
  tbc = table(f)
  tbc=sort(tbc,decreasing=F)
  tbcDf = as.data.frame(tbc)
  sums = 0
  
  for(i in 1:nrow(tbcDf)){
    sums = sums + tbcDf$Freq[i]
    if(sums >= CPercent){
      a <- as.character(tbcDf$f[1:i])
      break
    }
  }
  f<-as.character(f)
  f[(f %in% a)] <- "Others" 
  f <- as.factor(f)
  return (f) 
}

# Create OUT_DIR
dir.create(file.path(OUT_DIR), recursive=TRUE, showWarnings=FALSE)

# Inverse document frequency
if(OUT_IDF){
  idf_dt <- data.table::copy(dt)
  idf_dt[, (cols) := lapply(.SD, idfDeep), .SDcols = cols]
  write.table(idf_dt, file = paste(OUT_DIR, OUT_ID, 'idf.csv', sep=''), sep=',', row.names = FALSE)
}

# PCP + One Hot: idapplication, idoperator, idpartner, idcampaign, country
# One Hot: idaffmanager, idbrowser, idverticaltype, regioncontinent, accmanager
# IDF: city
if(OUT_PCP){
  # sapply(dt, function(x){length(levels(x))})
  pcp_dt <- data.table::copy(dt)
  
  # PCP columns
  pcp_cols <- c('idapplication', 'idoperator', 'idpartner', 'idcampaign', 'country')
  pcp_dt[, (pcp_cols) := lapply(.SD, PCP), .SDcols = pcp_cols]
  # sapply(pcp_dt, function(x){length(levels(as.factor(x)))})
  
  # One hot columns
  one_hot_cols <- c('idapplication', 'idoperator', 'idaffmanager', 'idbrowser', 'idpartner', 'idcampaign', 'idverticaltype', 
                    'regioncontinent', 'country', 'accmanager')
  pcp_dt <- as.data.table(dummy.data.frame(pcp_dt, names=one_hot_cols, sep='_'))
  
  # IDF column(s)
  idf_cols <- c('city')
  pcp_dt[, (idf_cols) := lapply(.SD, idfDeep), .SDcols = idf_cols]
  
  write.table(pcp_dt, file=paste(OUT_DIR, OUT_ID, 'pcp.csv', sep=''), sep=',', row.names = FALSE)
}