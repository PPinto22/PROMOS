library(rminer)
#library(rworldmap)
library(dtplyr)
library(countrycode)
library(jsonlite)
#library(RCurl)
#library(RMOA) 
library(dplyr)

#library(gramEvol)
library(doParallel)
library(DMwR) 
library(ROSE)
library(data.table)
library(caTools)
set.seed(123456789) 

option_list = list(
  make_option(c("-w","--workingdir"), type="character", default=".", help="working directory", metavar="DIR"),
  make_option(c("-s","--sales"), type="character", default="sales.json", help="path to the sales file", metavar="FILE"),
  make_option(c("-r","--redirects"), type="character", default="redis.json", help="path to the redirects file", metavar="FILE"),
  make_option(c("-o","--out"), type="character", default = "treated.json", help="out file", metavar = "FILE")
);
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

workingdir <- opt$workingdir
PathToFileS <- opt$sales
PathToFileR <- opt$redirects
out_file <- opt$out
dir.create(file.path(workingdir), recursive=TRUE, showWarnings=FALSE)
setwd(workingdir)

modFilterDates = "Week" #set the type of data you want (day, week....)

# get_os <- function(){
#   sysinf <- Sys.info()
#   if (!is.null(sysinf)){
#     os <- sysinf['sysname']
#     if (os == 'Darwin')
#       os <- "osx"
#   } else { ## mystery machine
#     os <- .Platform$OS.type
#     if (grepl("^darwin", R.version$os))
#       os <- "osx"
#     if (grepl("linux-gnu", R.version$os))
#       os <- "linux"
#   }
#   tolower(os)
#   return(os[[1]])
# }
# if(get_os()!="osx"){
#   setwd("C:/Users/luism/Desktop/Dataset2Semanas//")
#   PathToFileR = "ColetasPROMOSBEST-Redirects-2-Week-Best-2018-06-13"
#   PathToFileS = "ColetasPROMOSBEST-Sales-2-Week-Best-2018-06-13"
# }else{
#   setwd("Desktop/Datasets/TestesComParalelismo")
#   PathToFileR = "../ColetasPROMOS-Redirects-1-Week-Best.json"
#   PathToFileS = "../ColetasPROMOS-Sales-1-Week-Best.json"
# }


assign("ModeFilt", "BEST", envir = .GlobalEnv)
if(ModeFilt == "BEST"){
  RealisticSale = 0.01
}else{
  RealisticSale = 0.005
}
timeFit <- list()
timePredict <- list()
reditreated=data.frame()
target=""
regioncontinent=""
country_name=""
accmanager=""
PathToFileRName = paste("ColetasPROMOS",ModeFilt)





NC = 3

# apply these models "AdaHoeffdingOptionTree", "ASHoeffdingTree", "DecisionStump", "HoeffdingAdaptiveTree", "HoeffdingOptionTree", "HoeffdingTree", "LimAttHoeffdingTree", "RandomHoeffdingTree", "NaiveBayes",  "ADACC", "DACC", "OnlineAccuracyUpdatedEnsemble", "TemporallyAugmentedClassifier", "WeightedMajorityAlgorithm", "ActiveClassifier" ,"NaiveBayesMultinomial", "OCBoost", "OzaBoost","OzaBag","OzaBagAdwin","LeveragingBag", "OzaBagASHT" ,"OzaBoostAdwin", "ADACC", "OnlineAccuracyUpdatedEnsemble", "LimAttClassifier"
datatransform <- function(da = redi,
                          type = "redirect",
                          sa = sales, columnsKeep = c("id","date_added_utc","target","regioncontinent","idcampaign","idpartner","idverticaltype","idbrowser","idaffmanager","idapplication","idoperator","accmanager","country_name", "city")) {
  if (type == "redirect") {
    mergedtable <- mergesalesredi(redi = da,sales=sa)
    da$target <- ""
    
    da$regioncontinent <- ""
    
    da$regioncontinent <-
      countrycode(da$country_iso3166, "iso2c", "region")
    da$country_name <-
      countrycode(da$country_iso3166, "iso2c", "country.name")
    
    #da$hour <- as.numeric(da$hour)
    
    
    da$accmanager <- NA
    da$account_type[is.na(da$account_type)] <- "0"
    for (i in 1:nrow(da)) {
      # print(paste(i," transformation of ", nrow(da)))
      if (da$id[i] %in% c(mergedtable$idclick)) {
        da$target[i] <- "Sale"
      } else{
        da$target[i] <- "No Sale"
      }
      
      if (da$account_type[i] == "1") {
        da$accmanager[i] = 'WEBMASTER'
      } else if (da$account_type[i] == "2") {
        da$accmanager[i] = 'MEDIA BUYER'
      }
      
      else if (da$account_type[i] == "3") {
        da$accmanager[i] = 'NETWORK'
      }
      else if (da$account_type[i] == "4") {
        da$accmanager[i] = 'APP DEVELOPER'
      }
      
      else if (da$account_type[i] == "5") {
        da$accmanager[i] = 'FACEBOOK MARKETER'
      }
      else if (da$account_type[i] == "6") {
        da$accmanager[i] = 'ADVERTISER'
      }
      else if (da$account_type[i] == "7") {
        da$accmanager[i] = 'WEBMASTER SENSITIVE'
      }
      else if (da$account_type[i] == "8") {
        da$accmanager[i] = 'MEDIA BUYER SENSITIVE'
      }
      else if (da$account_type[i] == "9") {
        da$accmanager[i] = 'NETWORK SENSITIVE'
      }
      else if (da$account_type[i] == "10") {
        da$accmanager[i] = 'APP DEVELOPER SENSITIVE'
      }
      else if (da$account_type[i] == "11") {
        da$accmanager[i] = 'FACEBOOK MARKETER SENSITIVE'
      }
      else if (da$account_type[i] == "12") {
        da$accmanager[i] = 'ADVERTISER SENSITIVE'
      }
      else
      {
        da$accmanager[i] = 'Others'
      }
      
      
      
      
    }
    
    rmcols<-""
    for (cols in names(da)) {
      if(cols == "date_added_utc")next;
      if(!is.integer(da[,cols]) && !is.numeric(da[,cols]) && !is.character(da[,cols])){
        rmcols<-c(rmcols,cols)
      }else{
        if(is.character(da[,cols])){
          
          da[which(is.na((da[,cols]))),cols]<-"Others"
        }else if (is.numeric(da[,cols])){
          
          da[which(is.na((da[,cols]))),cols]<-0
        }
        
        
        
      }
      if(cols %in% c("ip", "fallback_campaign","browser_version", "identifier","afc_uniques","hour")){
        rmcols<-c(rmcols,cols)
      }
    }
    
    da<-da[,-which(names(da) %in% rmcols)]
    da<-da[,which(names(da) %in% columnsKeep)]
    #print("updating facts")
    
    #upFact(x=da)
    
    return(da)
  }
  
  
  if (type == "sale") {
    
    
    RealSales <- mergesalesredi(redi = get("redi"),sales=da)
    
    da<-checkDates(redi= get("redi"), sales= da) # this is because of old data collection still needed to be reduced to match the interval of the redirect data.
    
    #
    da<-removeDuples(sales = da,tablemerged = RealSales)
    da<-filterMode(data = da, modeFilter = get("ModeFilt"))
    da$target = "Sale"
    da$regioncontinent <-countrycode(da$country_iso3166, "iso2c", "region")
    da$country_name <-countrycode(da$country_iso3166, "iso2c", "country.name")
    da$accmanager <- NA
    da$account_type[is.na(da$account_type)] <- "0"
    for (i in 1:nrow(da)) {
      # print(paste(i," transformation of ", nrow(da)))
      
      
      if (da$account_type[i] == "1") {
        da$accmanager[i] = 'WEBMASTER'
      } else if (da$account_type[i] == "2") {
        da$accmanager[i] = 'MEDIA BUYER'
      }
      
      else if (da$account_type[i] == "3") {
        da$accmanager[i] = 'NETWORK'
      }
      else if (da$account_type[i] == "4") {
        da$accmanager[i] = 'APP DEVELOPER'
      }
      
      else if (da$account_type[i] == "5") {
        da$accmanager[i] = 'FACEBOOK MARKETER'
      }
      else if (da$account_type[i] == "6") {
        da$accmanager[i] = 'ADVERTISER'
      }
      else if (da$account_type[i] == "7") {
        da$accmanager[i] = 'WEBMASTER SENSITIVE'
      }
      else if (da$account_type[i] == "8") {
        da$accmanager[i] = 'MEDIA BUYER SENSITIVE'
      }
      else if (da$account_type[i] == "9") {
        da$accmanager[i] = 'NETWORK SENSITIVE'
      }
      else if (da$account_type[i] == "10") {
        da$accmanager[i] = 'APP DEVELOPER SENSITIVE'
      }
      else if (da$account_type[i] == "11") {
        da$accmanager[i] = 'FACEBOOK MARKETER SENSITIVE'
      }
      else if (da$account_type[i] == "12") {
        da$accmanager[i] = 'ADVERTISER SENSITIVE'
      }
      else
      {
        da$accmanager[i] = 'Others'
      }
      
      
      
      
    }
    da$account_type <- as.numeric(da$account_type)
    da$account_type[is.na(da$account_type)] <- 0
    rmcols<-""
    for (cols in names(da)) {
      if(cols == "date_added_utc")next;
      if(!is.integer(da[,cols]) && !is.numeric(da[,cols]) && !is.character(da[,cols])){
        rmcols<-c(rmcols,cols)
      }else{
        if(is.character(da[,cols])){
          
          da[which(is.na((da[,cols]))),cols]<-"Others"
        }else if (is.numeric(da[,cols])){
          
          da[which(is.na((da[,cols]))),cols]<-0
        }
        
        
        
      }
      if(cols %in% c("ip", "fallback_campaign","browser_version", "identifier","afc_uniques","hour")){
        rmcols<-c(rmcols,cols)
      }
    }
    
    da<-da[,-which(names(da) %in% rmcols)]
    da<-da[,which(names(da) %in% columnsKeep)]
    return(da)
  }
  
  
  
}


#update factors
upFact <- function(x) {
  cont = 1
  
  for (cols in colnames(x)) {
    a<-list()
    print(cols)
    
    
    
    
    a<-((x[,cols]))
    
    do.call("<<-",list(cols,unique(c(get(cols),unique(a)))))
    
    
  }
  
  
  
  
  
  
  
  
  
  cont = cont + 1
}




resetLevels <- function() {
  
  rm(list = colnames(redi))
}



#merge data function inner joins sales with redirects.
mergesalesredi<-function(redi, sales){
  #getSales()
  sales$idsales=sales$id
  #getRedirects()
  #sales<-sales[ , -which(names(sales) %in% c("http_headers", "ios_idfa","android_id", "_messid","_popsid", "user_agent","android_advertiser_id", "id"))]
  #redi<-redi[ , -which(names(redi) %in% c("http_headers"))]
  
  mergetable<-merge(sales,redi, by.x="idclick", by.y="id")
  
  return (mergetable)
}




exportTreatedData=function(x = 2,mod = modFilterDates, modef = ModeFilt, reditreated = reditreated){
  library(lubridate)
  reditreated$date_added_utc = as.POSIXct(reditreated$date_added_utc, format = "%Y-%m-%d %H:%M:%S")
  
  
  if (mod == "Week"){
    lowerRedi = min(reditreated$date_added_utc)
    UpperRedi = lowerRedi + (x*60*60*6) # add  6  hours
    I = which(reditreated$date_added_utc >= lowerRedi &
                reditreated$date_added_utc <= UpperRedi)
    rediT = reditreated[I, ]
    return (rediT)
  }else{
    return (NA)
  }
  
}




#Sain Project
#merge the redirects that made into a sale. For that we merged the table before with the actual sales and merge again
#with the redirects classing them as sale and not sale...



#This function has streaming capabilities for online streaming and offline streaming....

updates=function(W,H,N,S){ return ( ceiling( (N-(W+H))/S) ) }

checkDates = function(redi, sales) {
  
  redi$date_added_utc = as.POSIXct(redi$date_added_utc, format = "%Y-%m-%d %H:%M:%S")
  sales$date_added_utc = as.POSIXct(sales$date_added_utc, format = "%Y-%m-%d %H:%M:%S")
  
  date_r_min = min(redi$date_added_utc)
  date_r_max = max(redi$date_added_utc)
  I = which(sales$date_added_utc >= date_r_min &
              sales$date_added_utc <= date_r_max)
  d2 = sales[I, ]
  if(nrow(d2)>0){
    return(d2)
  }else{
    return(sales)
  }
  
  
  
  
  
}



removeDuples<-function(sales = sales , tablemerged = RealSales){
  I = which(!(sales$idclick %in% tablemerged$idclick))
  d2 = sales[I,]
  return (d2)
}

filterMode <- function(data, modeFilter = "Test") {
  if (modeFilter == "Test" || modeFilter == "TEST") {
    newData <-
      data[which(data$flag_test_mode == 1 | data$flag_test_mode == 3), ]
    
  } else {
    newData <-
      data[which(data$flag_test_mode == 0 | data$flag_test_mode == 2), ]
  }
  return (newData)
}

# 
# for (cols in colnames(redi)) {
#   do.call("assign",list(cols,""))
#   
# }

rbindFill.data.table <- function(master, newTable)  {
  # Append newTable to master
  
  # assign to Master
  #-----------------#
  # identify columns missing
  colMisng     <- setdiff(names(newTable), names(master))
  
  # if there are no columns missing, move on to next part
  if (!identical(colMisng, character(0)))  {
    # identify class of each
    colMisng.cls <- sapply(colMisng, function(x) class(newTable[[x]]))
    
    # assign to each column value of NA with appropriate class 
    master[ , eval(colMisng) := lapply(colMisng.cls, function(cc) as(NA, cc))]
  }
  
  # assign to newTable
  #-----------------#
  # identify columns missing
  colMisng     <- setdiff(names(master), names(newTable))
  
  # if there are no columns missing, move on to next part
  if (!identical(colMisng, character(0)))  {
    # identify class of each
    colMisng.cls <- sapply(colMisng, function(x) class(master[[x]]))
    
    # assign to each column value of NA with appropriate class 
    newTable[ , eval(colMisng) := lapply(colMisng.cls, function(cc) as(NA, cc))]
  }
  
  # reorder columns to avoid warning about ordering
  #-----------------#
  colOrdering <- colOrderingByOtherCol(newTable, names(master))
  setcolorder(newTable,  colOrdering)
  
  # rbind them! 
  #-----------------#
  rbind(master, newTable)
}



rbind.match.columns <- function(input1, input2) {
  n.input1 <- ncol(input1)
  n.input2 <- ncol(input2)
  
  if (n.input2 < n.input1) {
    TF.names <- which(names(input2) %in% names(input1))
    column.names <- names(input2[, TF.names])
  } else {
    TF.names <- which(names(input1) %in% names(input2))
    column.names <- names(input1[, TF.names])
  }
  
  return(rbind(input1[, column.names], input2[, column.names]))
}

sales = data.frame()
count = 1
stream_in(file(PathToFileS),handler = function(x){
  
  x=x[,-which(colnames(x) %in% c("http_headers","ip", "fallback_campaign","browser_version", "identifier","afc_uniques","hour","user_agent","_id"))]
  
  if(count == 1){
    print(get('count'))
    assign("sales", rbind(sales,x),envir = .GlobalEnv)
    assign('count', count+1, envir = .GlobalEnv)
  }else{
    #colnames(sales)<-colnames(x)
    
    d = rbind.match.columns(input1=get('sales'),input2=x)
    print(nrow(d))
    assign("sales", d ,envir = .GlobalEnv)
    
    
  }
}
,pagesize = 500)
count = 1
redi = data.frame()
stream_in(file(PathToFileR), handler = function(x){
  x<-datatransform(x, "redirect", sa = get("sales") ) 
  
  assign("redi", rbind(redi,x), envir = .GlobalEnv)
}, pagesize = 500)


salesT = data.frame()
stream_in(file(PathToFileS), handler = function(x){
  
  x<-datatransform(da = x, type="sale")
  
  assign("salesT", rbind(salesT,x),envir = .GlobalEnv)
}



,pagesize = 10000)

if(out_file == ''){ 
  FileTreated = paste(PathToFileRName,"Treated",Sys.Date(),".json", sep="") 
} else{
  FileTreated = out_file
}
dir.create(file.path(dirname(out_file)), recursive=TRUE, showWarnings=FALSE)
if(file.exists(FileTreated)){
  assign("reditreated", stream_in(file(FileTreated)), envir = .GlobalEnv)
  
  
}else{
  
  
  
  
  
  redi$date_added_utc = as.POSIXct(redi$date_added_utc, format = "%Y-%m-%d %H:%M:%S")
  salesT$date_added_utc = as.POSIXct(sales$date_added_utc, format = "%Y-%m-%d %H:%M:%S")
  
  reditreated<-rbind(redi,salesT)
  
  reditreated<-reditreated[order(reditreated$date_added_utc),]
  reditreated2 = exportTreatedData(reditreated = reditreated)
  
  stream_out(reditreated2, con=file(FileTreated))
}
