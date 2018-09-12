library(mongolite)
library(jsonlite)
library(doParallel)
library(optparse)

option_list = list(
  make_option(c("-w","--workingdir"), type="character", default=".", help="working directory", metavar="DIR"),
  make_option(c("-s","--sales"), type="character", default="", help="path where to save the extracted sales", metavar="FILE"),
  make_option(c("-r","--redirects"), type="character", default="", help="path where to save the extracted redirects", metavar="FILE"),
  make_option(c("-t","--time"), type="double", default = 6, help="collection time duration in hours", metavar = "H")
);
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

workingdir <- opt$workingdir
sales_file <- opt$sales
redis_file <- opt$redirects
exec_time <- opt$time
remove(option_list, opt_parser, opt)

dir.create(file.path(workingdir), recursive=TRUE, showWarnings=FALSE)
setwd(workingdir)

#GLOBAL VARIABLES SETTINGS
LD = NA #Lower Date Sales
UD = NA #Upper Date Sales
LR = NA #Lower Date Redirects
UR = NA #Upper Date Redirects
tksR = 0 #contador de ticks Redirects
tksS = 0 #contador de ticks Sales
FileNameUR = "date_uppR.txt"
FileNameLR = "date_lowR.txt"
FileNameLD = "date_lowS.txt"
FileNameUD = "date_uppS.txt"
waitTimeFetchingR = 0 #1*60 secs --> 1 minute
waitTimeFetchingS = 0 #1*60 secs --> 1 minute
TimeLapse = 1 #hours after the interval of UD
NR =  2500 #Number of redirects to fetch in the stream
NS = 350 #Number of sales to fetch in the stream
NRInitial = 50 #NR Base dos pedidos
NSInitial = 50 #NS Base dos pedidos
MDR = 300 * (2 ^ 20) #Number,in bytes, corresponding a 5GB of redirects
MDFR = 1000 * (2 ^ 20)# 625*2^20/1024 # FIFO SIZE REDIRECTS IGNORAR
MDFS = 500 * (2 ^ 20)#375*2^20/1024 #FIFO SIZE SALES IGNORAR
LF = 50 * 2 ^ 20 # log size file
MDS = 300 * (2 ^ 20) #1000*2^20 #Number, in bytes, corresponding a 3GB of sales
WriteFile = T # If the size is reached in MDR, this is make it to stop writing redirects to the static file
WriteSales = T # If the size is reached in MDS, this is make it to stop writing sales to the static file
STR = 0 #System sleep time for redirects
STS = 0 #System sleep time for sales
NAMEDB = "ColetasPROMOSBESTPEDRO" #Name of the database inside mongoDB
# NAMEREDI = "redirects" # name for the collection inside mongoDB for redirects (dynamic)
# NAMESALES = "sales" # name for the collection inside mongodb for sales (dynamic)
STATICREDI = "RediStaticV4PEDRO" # name for the static data collection inside mongodb (redirects)
STATICSALES = "SalesStaticV4PEDRO" # name for the static data collection inside mongodb (sales)
NAMELOG = "LogsBEST" #log collection name
#Javascript file to create the settings inside the database mongo
DBCREATIONFILENAME = "jsdatabase2PEDRO.js"
# COLLECTIONREDICREATIONFILENAME = "collectionredififo.js" #Deprecated
# COLLECTIONSALESCREATIONFILENAME = "collectionsalesfifo.js" #Deprecated
COLLECTIONLOGSCREATIONFILENAME = "LogFifoV4PEDRO.js"
COLLECTIONREDISTATICCREATIONFILENAME = "collectionredistaticV4PEDRO.js"
COLLECTIONSALESSTATICCREATIONFILENAME = "collectionsalesstaticV4PEDRO.js"

InicialData = Sys.time() 
TimeStop = exec_time * 60 * 60
FinalData = InicialData + TimeStop #when stops the code

thresholdRGLOBAL<- 70 #Folga de 70% de Redirects
thresholdSGLOBAL<- 70 #Folga de 70% de Sales
C <-
  as.list(.GlobalEnv) # saves all variables in a list to add to log files
A <- as.data.frame(C)

filllogs <- function(DataLogs, c = C) {
  X <- lsf.str()
  Y <- as.vector(X)
  for (cols in colnames(A)) {
    if (cols %in% Y) {
      next
    }
    DataLogs[, cols] <- get(cols)
  }
  return(DataLogs)
}

DATALOG <- filllogs(A) #creation of the LOG TABLE to mongo
#END OF GLOBAL SETTINGS

#This fucntion will create the files needed to execute in the shell with R, which will create the database and the collections with their sizes specified in the global settings

init <- function() {
  if (file.exists(FileNameUR))
    file.remove(FileNameUR)
  if (file.exists(FileNameLR))
    file.remove(FileNameLR)
  if (file.exists(FileNameUD))
    file.remove(FileNameUD)
  if (file.exists(FileNameLD))
    file.remove(FileNameLD)
  if (file.exists("statusRedi.txt"))
    file.remove("statusRedi.txt")
  if (file.exists("Timefull.txt"))
    file.remove("Timefull.txt")
  if (file.exists("statussales.txt"))
    file.remove("statussales.txt")
  createdb()
  # createCollectionfifo() # redirects
  # createCollectionfifo( namefifo = NAMESALES ,db=NAMEDB, namefilejs = COLLECTIONSALESCREATIONFILENAME, size = MDFS ) #sales
  createCollectionfifo(
    namefifo = NAMELOG ,
    db = NAMEDB,
    namefilejs = COLLECTIONLOGSCREATIONFILENAME,
    size = LF
  ) #log
  createCollectionstatic() # redirect static
  createCollectionstatic(
    namestatic = STATICSALES,
    db = NAMEDB,
    namefilejs = COLLECTIONSALESSTATICCREATIONFILENAME,
    size = MDS
  )
}

createdb <- function(Namedb = NAMEDB, nameFile = DBCREATIONFILENAME) {
  #database creation
  
  mongoC <-
    "var MongoClient = require('mongodb').MongoClient;var assert = require('assert');var Db = require('mongodb').Db,Server = require('mongodb').Server,
  ReplSetServers = require('mongodb').ReplSetServers,
  ObjectID = require('mongodb').ObjectID,
  Binary = require('mongodb').Binary,
  GridStore = require('mongodb').GridStore,
  Grid = require('mongodb').Grid,
  Code = require('mongodb').Code;"
  url = paste("var url = \"mongodb://localhost:27017/", Namedb, "\";", sep =
                "")
  code = paste(
    "MongoClient.connect(url, function(err, db) {
    
    db.dropDatabase(function(err, result) {
    if (err) throw err;
    console.log(result)
    assert.equal(null, err);
    setTimeout(function() {
    db.admin().listDatabases(function(err, dbs)
    {dbs = dbs.databases;
    var found = false;
    for(var i = 0; i < dbs.length; i++) {
    if(dbs[i].name == '",
    Namedb,
    "')
    found = true;
    }
    if(process.env['JENKINS'] == null)
    assert.equal(false, found);
    db.close();
    });
    }, 2000);
    });
});" ,
sep = ""
)
  
  write(mongoC, file = nameFile, sep = "\n")
  write(url,
        file = nameFile,
        sep = "\n",
        append = T)
  write(code,
        file = nameFile,
        sep = "\n",
        append = T)
  system(paste("node", nameFile))
  }

createCollectionfifo <-
  function(namefifo = NAMEREDI ,
           db = NAMEDB,
           namefilejs = COLLECTIONREDICREATIONFILENAME,
           size = MDR) {
    mongoC <- "var MongoClient = require('mongodb').MongoClient;"
    url = paste("var url = \"mongodb://localhost:27017/", db, "\";", sep =
                  "")
    
    code = paste(
      " MongoClient.connect(url, function(err, db) { if (err) throw err; db.createCollection(\"",
      namefifo,
      " \", {\"capped\" : true, \"size\" :",
      size,
      "}, function(err, res) {if (err) throw err;console.log(\"Collection Created!\");db.close();}); });",
      sep = ""
    )
    
    write(mongoC, file = namefilejs, sep = "\n")
    write(url,
          file = namefilejs,
          sep = "\n",
          append = T)
    write(code,
          file = namefilejs,
          sep = "\n",
          append = T)
    system(paste("node", namefilejs))
  }

createCollectionstatic <-
  function(namestatic =  STATICREDI ,
           db = NAMEDB,
           namefilejs = COLLECTIONREDISTATICCREATIONFILENAME,
           size = MDFR) {
    mongoC <- "var MongoClient = require('mongodb').MongoClient;"
    url = paste("var url = \"mongodb://localhost:27017/", db, "\";", sep =
                  "")
    
    code = paste(
      " MongoClient.connect(url, function(err, db) { if (err) throw err; db.createCollection(\"",
      namestatic,
      " \", {\"capped\" : false, \"size\" :",
      size,
      "}, function(err, res) {if (err) throw err;console.log(\"Collection Created!\");db.close();}); });",
      sep = ""
    )
    
    write(mongoC, file = namefilejs, sep = "\n")
    write(url,
          file = namefilejs,
          sep = "\n",
          append = T)
    write(code,
          file = namefilejs,
          sep = "\n",
          append = T)
    system(paste("node", namefilejs))
  }
filterContinents <- function(data, continentFilter = "Europe") {
  library(countrycode)
  data$continent <-
    countrycode(
      sourcevar = data$country_iso3166 ,
      origin = "iso2c",
      destination = "continent"
    )
  newData <- data[which(data$continent %in% continentFilter), ]
  return (newData)
}
filterCountries <- function(data, Country = "Portugal") {
  library(countrycode)
  data$Country <-
    countrycode(
      sourcevar = data$country_iso3166 ,
      origin = "iso2c",
      destination = "country.name"
    )
  newData <- data[which(data$Country %in% Country), ]
  return (newData)
}
getRedirects <-
  function(link = "https://uni.rtman.net:1639/redirects/",
           N = NR,
           DataLog = DATALOG,
           needFilter = T,
           ContinentF = T,
           CountryF = T,
           CountryName = "Portugal",
           typeFilter = "Europe" ,
           bestmodeFilter = "Both", NRI = NRInitial) {
    print("redirects")
    link <- paste(link, N, sep = "")
    DataLog$InitialRequestTime = Sys.time()
    DataLog$IDay <- format(Sys.time(), "%d")
    DataLog$IMonth <- format(Sys.time(), "%m")
    DataLog$IYear <- format(Sys.time(), "%Y")
    DataLog$IHour <- format(Sys.time(), "%H")
    DataLog$IMinute <- format(Sys.time(), "%M")
    DataLog$ISecond <- format(Sys.time(), "%S")
    #mongod = mongo(collection = NAMEREDI, db = NAMEDB)
    measure <-
      system.time(d <-
                    tryCatch(
                      d <-
                        stream_in(url(link)),
                      silent = F,
                      error = function(e) {
                        d <- conditionMessage(e)
                      }
                    ))
    #t<-try(mongod$insert(d),silent = F)
    logs = mongo(collection = NAMELOG, db = NAMEDB)
    # print(d)
    
    DataLog$Request = "Redirects"
    DataLog$RequestDuration <- measure[[3]]
    if (is.data.frame(d)) {
      DataLog$Number_Before_Filter = nrow(d)
      DataLog$WasFilterUsed = needFilter
      
      
      
      
      DataLog$ModeFilter = bestmodeFilter
      
      
      d$date_added_utc = as.POSIXct(d$date_added_utc, format = "%Y-%m-%dT%H:%M:%SZ")
      DataLog$NumberofIsVisit1 <- length(which(d$is_visit == 1))
      
      d <- d[which(d$is_visit == 1), ]
      if (needFilter) {
        if (ContinentF) {
          print("Continent Filter is applied")
          print(typeFilter)
          d <- filterContinents(data = d, continentFilter = typeFilter)
          
          
          if (length(typeFilter) > 1) {
            types_as_string = ""
            for (j in 1:length(typeFilter)) {
              types_as_string = paste(types_as_string, as.character(typeFilter[[j]]), sep =
                                        ";")
            }
            DataLog$TypeFilter = types_as_string
          } else{
            DataLog$TypeFilter = typeFilter
          }
        }
        
        
        
        if (CountryF) {
          print("Country Filter Applied")
          print(CountryName)
          if (length(CountryName) > 1) {
            types_as_string2 = ""
            for (j in 1:length(CountryName)) {
              types_as_string2 = paste(types_as_string2,
                                       "| ",
                                       as.character(CountryName[[j]]),
                                       sep = ";")
            }
            DataLog$TypeFilterCountry = types_as_string2
          } else{
            DataLog$TypeFilterCountry = CountryName
          }
          d <- filterCountries(data = d, Country = CountryName)
          
          
        }
        if (bestmodeFilter == "Best") {
          print("Best Mode ")
          d <- filterMode(data = d, modeFilter = "Best")
          
        } else if (bestmodeFilter == "Test") {
          print("Test Mode")
          d <- filterMode(data = d)
        } else{
          print("Both modes selected")
        }
        DataLog$NumberWithFilter = nrow(d)
        print(paste("Rows after filter", nrow(d), sep = " "))
        
        if(nrow(d)>=NRI){
          print("Redirects above the base value, sampling....")
          LinhasSamples<-sample(1:nrow(d), size = NRI)
          d<-d[LinhasSamples,]
          
          verifyTicksR(dataset=d)
        }else{
          print("Redirects below the base value")
          
          verifyTicksR(dataset=d)
        }
      }
      DataLog$EndRequestTime = Sys.time()
      DataLog$Day <- format(Sys.time(), "%d")
      DataLog$Month <- format(Sys.time(), "%m")
      DataLog$Year <- format(Sys.time(), "%Y")
      DataLog$Hour <- format(Sys.time(), "%H")
      DataLog$Minute <- format(Sys.time(), "%M")
      DataLog$Second <- format(Sys.time(), "%S")
      logT <- try(logs$insert(DataLog))
    } else {
      DataLog$Number = 0
      DataLog$EndRequestTime = Sys.time()
      DataLog$Day <- format(Sys.time(), "%d")
      DataLog$Month <- format(Sys.time(), "%m")
      DataLog$Year <- format(Sys.time(), "%Y")
      DataLog$Hour <- format(Sys.time(), "%H")
      DataLog$Minute <- format(Sys.time(), "%M")
      DataLog$Second <- format(Sys.time(), "%S")
      
      
      #print(DataLog)
      
    }
    
    Sys.sleep(waitTimeFetchingR)
    return(d)
  }
getSales <-
  function(link = "https://uni.rtman.net:1639/sales/",
           N = NS,
           windowProcess = 500,
           DataLog = DATALOG,
           needFilter = T,
           typeFilter = "Europe",
           ContinentF = T,
           CountryF = T,
           CountryName = "Portugal", NSI = NSInitial) {
    print("sales")
    
    link <- paste(link, N, sep = "")
    DataLog$InitialRequestTime = Sys.time()
    DataLog$IDay <- format(Sys.time(), "%d")
    DataLog$IMonth <- format(Sys.time(), "%m")
    DataLog$IYear <- format(Sys.time(), "%Y")
    DataLog$IHour <- format(Sys.time(), "%H")
    DataLog$IMinute <- format(Sys.time(), "%M")
    DataLog$ISecond <- format(Sys.time(), "%S")
    measure <-
      system.time(ee <-
                    tryCatch(
                      d <-
                        stream_in(url(link), pagesize = windowProcess),
                      error = function(e) {
                        d <- conditionMessage(e)
                      }
                    ))
    #mongod = mongo(collection = NAMESALES, db = NAMEDB)
    # t<-try(mongod$insert(d),silent = T)
    logs = mongo(collection = NAMELOG, db = NAMEDB)
    
    DataLog$Request = "Sales"
    DataLog$RequestDuration <- measure[[3]]
    if (is.data.frame(ee)) {
      DataLog$Number_Before_Filter = nrow(ee)
      DataLog$WasFilterUsed = needFilter
      
      ee$date_added_utc = as.POSIXct(ee$date_added_utc, format = "%Y-%m-%dT%H:%M:%SZ")
      if (needFilter) {
        print("Filter is applied for SALES")
        if (ContinentF) {
          print("Continent Filter is applied")
          print(typeFilter)
          ee <- filterContinents(data = ee, continentFilter = typeFilter)
          if (length(typeFilter) > 1) {
            types_as_string = ""
            for (j in 1:length(typeFilter)) {
              types_as_string = paste(types_as_string,
                                      "| ",
                                      as.character(typeFilter[[j]]),
                                      sep = ";")
            }
            DataLog$TypeFilter = types_as_string
          } else{
            DataLog$TypeFilter = typeFilter
          }
        }
        if (CountryF) {
          print("Country Filter Applied")
          print(CountryName)
          if (length(CountryName) > 1) {
            types_as_string2 = ""
            for (j in 1:length(CountryName)) {
              types_as_string2 = paste(types_as_string2,
                                       "| ",
                                       as.character(CountryName[[j]]),
                                       sep = ";")
            }
            DataLog$TypeFilterCountry = types_as_string2
          } else{
            DataLog$TypeFilterCountry = CountryName
          }
          ee <- filterCountries(data = ee, Country = CountryName)
          
          
        }
        DataLog$NumberWithFilter = nrow(ee)
        
        print(paste("Sales after filter", nrow(ee), sep = " "))
        if(nrow(ee)>=NSI){
          print("Sales above the base value, sampling....")
          LinhasSamples<-sample(1:nrow(ee), size = NSI)
          ee<-ee[LinhasSamples,]
          verifyTicksS(dataset=ee)
        }else{
          print("Sales below the base value")
          verifyTicksS(dataset=ee)
        }
        
      }
      DataLog$EndRequestTime = Sys.time()
      DataLog$Day <- format(Sys.time(), "%d")
      DataLog$Month <- format(Sys.time(), "%m")
      DataLog$Year <- format(Sys.time(), "%Y")
      DataLog$Hour <- format(Sys.time(), "%H")
      DataLog$Minute <- format(Sys.time(), "%M")
      DataLog$Second <- format(Sys.time(), "%S")
      logT <- try(logs$insert(DataLog))
    }else{
      DataLog$Number = 0
      DataLog$EndRequestTime = Sys.time()
      DataLog$Day <- format(Sys.time(), "%d")
      DataLog$Month <- format(Sys.time(), "%m")
      DataLog$Year <- format(Sys.time(), "%Y")
      DataLog$Hour <- format(Sys.time(), "%H")
      DataLog$Minute <- format(Sys.time(), "%M")
      DataLog$Second <- format(Sys.time(), "%S")
    }
    
    Sys.sleep(waitTimeFetchingS)
    return(ee)
  }


filterMode <- function(data, modeFilter = "test") {
  if (modeFilter == "test" || modeFilter == "Test") {
    newData <-
      data[which(data$flag_test_mode == 1 | data$flag_test_mode == 3), ]
  } else {
    newData <-
      data[which(data$flag_test_mode == 0 | data$flag_test_mode == 2), ]
  }
  return (newData)
}
staticdatafill <-
  function(collection,
           dataatthattime,
           sizermax = MDR ,
           type = "redirects",
           FilenameLow = FileNameLR,
           FilenameUpp = FileNameUR) {
    if (is.null(collection$info()$stats$size)) {
      f <-
        try(collection$insert(dataatthattime), silent = T)
      ;insertAgain = F
    } else{
      insertAgain = T
    }
    if (!file.exists(FileNameLR) &&
        type == "sales") {
      lowerDate = min(dataatthattime$date_added_utc)
      write.table(data.frame(low = lowerDate),
                  file = FilenameLow,
                  row.names = FALSE)
    }
    print(nrow(dataatthattime))
    #logT<-try(collection$insert(dataatthattime))
    if (type == "redirects") {
      lowerDate = min(dataatthattime$date_added_utc)
      upperDate = max(dataatthattime$date_added_utc)
      assign("LR", lowerDate,  envir = .GlobalEnv)
      assign("UR", upperDate,  envir = .GlobalEnv)
      #write.csV3(as.data.frame(LD), "lowerdate.csv", row.names = F)
      #write.csV3(as.data.frame(UD), "Upperdate.csv", row.names = F)
      if (!file.exists(FilenameLow)) {
        write.table(data.frame(low = lowerDate),
                    file = FilenameLow,
                    row.names = FALSE)
      }
      write.table(data.frame(upp = upperDate),
                  file = FilenameUpp,
                  row.names = FALSE)
    }
    
    logs = mongo(collection = NAMELOG, db = NAMEDB)
    logs$export(file(paste(NAMELOG, ".json", sep = "")))
    if (type == "redirects") {
      print(collection$info()$stats$size)
      
      if (collection$info()$stats$size >= sizermax) {
        assign("WriteFile", FALSE,  envir = .GlobalEnv)
        writeRedi <- F
        write.table(writeRedi,
                    "statusRedi.txt",
                    row.names = F,
                    col.names = F)
        write.table(Sys.time(),
                    "Timefull.txt",
                    row.names = F,
                    col.names = F)
        
        
        
      } else{
        if (insertAgain) {
          f <- try(collection$insert(dataatthattime), silent = T)
        } else{
          insertAgain = T
        }
        
        collection$export(file(paste(
          NAMEDB, type, ".json", sep = ""
        )))
      }
      
    } else if (type == "sales") {
      lowerDate = min(dataatthattime$date_added_utc)
      upperDate = max(dataatthattime$date_added_utc)
      assign("LD", lowerDate,  envir = .GlobalEnv)
      assign("UD", upperDate,  envir = .GlobalEnv)
      #write.csV3(as.data.frame(LD), "lowerdate.csv", row.names = F)
      #write.csV3(as.data.frame(UD), "Upperdate.csv", row.names = F)
      if (!file.exists(FilenameLow)) {
        write.table(data.frame(low = lowerDate),
                    file = FilenameLow,
                    row.names = FALSE)
      }
      write.table(data.frame(upp = upperDate),
                  file = FilenameUpp,
                  row.names = FALSE)
      
      if (file.exists("statusRedi.txt")) {
        estado = read.table("statusRedi.txt")
        print(paste("redirect state", estado))
        if (!estado) {
          if (file.exists("Timefull.txt")) {
            timefilled = read.table("Timefull.txt")
            datefull <- paste(timefilled$V1, timefilled$V2)
            diffs <- difftime(Sys.time(), datefull, units = "hour")
            if (diffs[[1]] >= TimeLapse) {
              Sa <- F
              write.table(Sa,
                          "statussales.txt",
                          row.names = F,
                          col.names = F)
              # f<-try(collection$insert(dataatthattime), silent = T)
              collection$export(file(paste(
                NAMEDB, type, ".json", sep = ""
              )))
            }
          }
        }
      }
      if (collection$info()$stats$size <= sizermax) {
        if (insertAgain) {
          f <- try(collection$insert(dataatthattime), silent = T)
        } else{
          insertAgain = T
        }
        collection$export(file(paste(
          "TestFilterPromos2", type, ".json", sep = ""
        )))
      }
      if (!file.exists("statussales.txt") ||
          collection$info()$stats$size >= sizermax) {
        collection$export(file(paste(
          NAMEDB, type, ".json", sep = ""
        )))
        
      }
      
    }
  }

dx <- function(d) {
  my_collection2 = mongo(collection = STATICREDI, db = NAMEDB)
  tryCatch(
    # assign("LISTOFIDSPERTICK_REDI",   LISTOFIDSPERTICK_REDI<-rbind(LISTOFIDSPERTICK_REDI,nrow(d)), envir = .GlobalEnv),
    # assign("tksR",tksR <- tksR +1, envir = .GlobalEnv),
    staticdatafill(collection = my_collection2, dataatthattime = d),
    
    error = function(e) {
      cat(conditionMessage(e))
    }
  )
  
  
}

dsales <- function(d) {
  my_collection2 = mongo(collection = STATICSALES , db = NAMEDB)
  tryCatch(
    #assign("LISTOFIDSPERTICK_SALES",  LISTOFIDSPERTICK_SALES<-rbind(LISTOFIDSPERTICK_SALES,nrow(d)), envir = .GlobalEnv),
    #assign("tksS",tksS <- tksS +1, envir = .GlobalEnv),
    staticdatafill(
      collection = my_collection2,
      dataatthattime = d,
      type = "sales",
      sizermax = MDS,
      FilenameLow = "date_lowS.txt",
      FilenameUpp = "date_uppS.txt"
    ),
    error = function(e) {
      cat(conditionMessage(e))
    }
  )
  
  
}

verifyTicksR<-function(dataset){
  
  #auto ajuste
  print("Auto Adjusting NR")
  ajustaFunctionR(idsFromTicksR = nrow(dataset), thresholdR = thresholdRGLOBAL)
  
  
}

verifyTicksS<-function(dataset){
  
  #auto ajuste
  print("Auto Adjusting NS")
  ajustaFunctionS(idsFromTicksS = nrow(dataset), thresholdS = thresholdSGLOBAL)
  
  
  
  
}



runs <-
  function(Filter = T,
           Best = "Both",
           Fcontinent = T,
           Fcountry = T,
           countryN = "Portugal",
           Continent = "Europe", Final = FinalData) {
    repeat {
      if(Sys.time() >= Final){
        print("Maximum time reached stopping data collecting")
        break
      } 
      #verifyTicksR()
      if (Filter) {
        d <-
          tryCatch(
            d <-
              getRedirects(
                needFilter = Filter,
                bestmodeFilter = Best,
                ContinentF = Fcontinent,
                CountryF = Fcountry,
                CountryName = countryN,
                typeFilter = Continent
              ),
            silent = F,
            error = function(e) {
              d <- conditionMessage(e)
            }
          )
        
      } else{
        d <-
          tryCatch(
            d <-
              getRedirects(needFilter = Filter),
            silent = F,
            error = function(e) {
              d <- conditionMessage(e)
            }
          )
        
      }
      
      if (is.data.frame(d) && nrow(d)>0 ) {
        dx(d)
        
      } else if (!is.data.frame(d)){
        print (d)
        logs = mongo(collection = NAMELOG , db = NAMEDB)
        DataLog <- as.data.frame(NA)
        
        DataLog <- filllogs(DataLogs = DataLog)
        DataLog$InitialRequestTime = Sys.time()
        DataLog$Request = "Redirects"
        DataLog$Number = 1
        DataLog$Day <- format(Sys.time(), "%d")
        DataLog$Month <- format(Sys.time(), "%m")
        DataLog$Year <- format(Sys.time(), "%Y")
        DataLog$Hour <- format(Sys.time(), "%H")
        DataLog$Minute <- format(Sys.time(), "%M")
        DataLog$Second <- format(Sys.time(), "%S")
        DataLog$error <- d
        logT <- try(logs$insert(DataLog))
        logs$export(file(paste(NAMELOG, ".json", sep = "")))
        
      }  
      
      
      #Sys.sleep(20)
      print(get("Break"))
      if (file.exists("statusRedi.txt")) {
        break
        
      }
      
    }
  }





runsales <-
  function(Filter = T,
           Fcontinent = T,
           Fcountry = T,
           countryN = "Portugal",
           Continent = "Europe", Final = FinalData) {
    repeat {
      # verifyTicksS()
      if(Sys.time() >= Final){
        print("Maximum time reached stopping data collecting")
        break
      } 
      
      if (Filter) {
        d <-
          tryCatch (
            d <-
              getSales(
                needFilter = Filter,
                typeFilter = Continent,
                ContinentF = Fcontinent,
                CountryF = Fcountry,
                CountryName = countryN
              ),
            silent = F,
            error = function(e) {
              d <- conditionMessage(e)
            }
          )
        
      } else{
        d <-
          tryCatch (
            d <-
              getSales(needFilter = F, typeFilter = Continent),
            silent = F,
            error = function(e) {
              d <- conditionMessage(e)
            }
          )
        
      }
      if (is.data.frame(d) && nrow(d)>0) {
        # d2 = checkDates(d)
        # if(!is.null(d2)){
        #   dsales(d2)
        # }else{
        #   dsales(d)
        # }
        dsales(d)
        
      } else if (!is.data.frame(d)){
        print (d)
        logs = mongo(collection = NAMELOG, db = NAMEDB)
        DataLog <- as.data.frame(NA)
        DataLog <- filllogs(DataLogs = DataLog)
        DataLog$InitialRequestTime = Sys.time()
        DataLog$Request = "Sales"
        DataLog$Number = 1
        DataLog$Day <- format(Sys.time(), "%d")
        DataLog$Month <- format(Sys.time(), "%m")
        DataLog$Year <- format(Sys.time(), "%Y")
        DataLog$Hour <- format(Sys.time(), "%H")
        DataLog$Minute <- format(Sys.time(), "%M")
        DataLog$Second <- format(Sys.time(), "%S")
        DataLog$error <- d
        logT <- try(logs$insert(DataLog))
        logs$export(file(paste(NAMELOG, ".json", sep = "")))
        
      }  
      
      
      
      #Sys.sleep(STS)
    }
    
  }
mergesalesredi <- function(redi, sales) {
  sales$idsales = sales$id #para evitar colunas como o mesmo nome depois.
  
  sales <-
    sales[,-which(
      names(sales) %in% c(
        "http_headers",
        "ios_idfa",
        "android_id",
        "_messid",
        "_popsid",
        "user_agent",
        "android_advertiser_id",
        "id"
      )
    )]
  redi <- redi[,-which(names(redi) %in% c("http_headers"))]
  
  mergetable <- merge(sales, redi, by.x = "idclick", by.y = "id")
  
  return (mergetable)
}


#run sales fetching function while processing a record at a time (testing if it is faster)
getSalesV2 <-
  function(link = "https://uni.rtman.net:1639/sales/",
           N = NS,
           windowProcess = 1,
           DataLog = DATALOG,
           type = "SalesTest") {
    print("sales")
    
    link <- paste(link, N, sep = "")
    ee <-
      tryCatch(
        d <-
          stream_in(url(link), pagesize = windowProcess),
        error = function(e) {
          d <- conditionMessage(e)
        }
      )
    #mongod = mongo(collection = NAMESALES, db = NAMEDB)
    # t<-try(mongod$insert(d),silent = T)
    logs = mongo(collection = NAMELOG, db = NAMEDB)
    
    DataLog$Request = "Sales"
    if (is.data.frame(ee)) {
      DataLog$Number = nrow(ee)
    }
    DataLog$Day <- format(Sys.time(), "%d")
    DataLog$Month <- format(Sys.time(), "%m")
    DataLog$Year <- format(Sys.time(), "%Y")
    DataLog$Hour <- format(Sys.time(), "%H")
    DataLog$Minute <- format(Sys.time(), "%M")
    DataLog$Second <- format(Sys.time(), "%S")
    logT <- try(logs$insert(DataLog))
    logs$export(file(paste(NAMELOG, ".json", sep = "")))
    my_collection2 = mongo(collection = STATICSALES , db = NAMEDB)
    if (is.data.frame(ee)) {
      my_collection2$insert(ee)
      
      my_collection2$export(file(paste(
        "TestFilterPromos2", type, ".json", sep = ""
      )))
    }
    return(ee)
  }

# Runsales for a certain time in Hours
runsalesV2 <- function(TimeRun = 1) {
  # if (file.exists("TimeFetchSalesStarted.txt")) file.remove("TimeFetchSalesStarted.txt");
  #
  #
  #
  # write.table(Sys.time(), "TimeFetchSalesStarted.txt", row.names = F, col.names = F)
  
  timeStart = Sys.time()
  
  
  repeat {
    diffs <- difftime(Sys.time(), timeStart, units = "hour")
    if (diffs[[1]] >= TimeRun) {
      break
      
    }
    d <-
      tryCatch (
        d <-
          getSalesV2(windowProcess = 100),
        silent = F,
        error = function(e) {
          d <- conditionMessage(e)
        }
      )
    if (is.data.frame(d)) {
      # d2 = dsales(d)
      # if(nrow(d2) > 0 && !is.null(d2)) {
      #   # in case sales data is referent to actual redirects
      #
      # }else{
      #   logs = mongo(collection = NAMELOG, db = NAMEDB)
      #   DataLog<-as.data.frame(NA)
      #   DataLog<-filllogs(DataLogs = DataLog)
      #   DataLog$Request = "Sales"
      #   DataLog$Number = 1
      #   DataLog$Day<-format(Sys.time(), "%d")
      #   DataLog$Month<-format(Sys.time(), "%m")
      #   DataLog$Year<-format(Sys.time(), "%Y")
      #   DataLog$Hour<-format(Sys.time(), "%H")
      #   DataLog$Minute<-format(Sys.time(), "%M")
      #   DataLog$Second<-format(Sys.time(), "%S")
      #   DataLog$error <- "No sale matching the interval of redirects"
      #   logT<-try(logs$insert(DataLog))
      #   logs$export(file(paste(NAMELOG,".json",sep="")))
      # }
      
    } else{
      print (d)
      logs = mongo(collection = NAMELOG, db = NAMEDB)
      DataLog <- as.data.frame(NA)
      DataLog <- filllogs(DataLogs = DataLog)
      DataLog$Request = "Sales"
      DataLog$Number = 1
      DataLog$Day <- format(Sys.time(), "%d")
      DataLog$Month <- format(Sys.time(), "%m")
      DataLog$Year <- format(Sys.time(), "%Y")
      DataLog$Hour <- format(Sys.time(), "%H")
      DataLog$Minute <- format(Sys.time(), "%M")
      DataLog$Second <- format(Sys.time(), "%S")
      DataLog$error <- d
      logT <- try(logs$insert(DataLog))
      logs$export(file(paste(NAMELOG, ".json", sep = "")))
    }
    
    
    #Sys.sleep(STS)
  }
  
}

#init()

# registerDoParallel(2)
assign("Break", FALSE, envir = .GlobalEnv)
print(get("Break"))



#Runs the program
#For multi continent fetch just do Continent=c("Europe","Asia") or else leave Continent=c("Europe")
#for BestMode we should write Test, Best or Both with the purpose of choosing a single filter, like
#I want all africa and asia records with both types... Or I want the same but test mode...
#Now added the country name filter sames as continents...
#The function now makes filter for sales, redirects (individually), by continent and country, or by country only or continent only.
RunProgram <-
  function(filter = T,
           FilterSales = T,
           bestMode = "Both",
           FilterContinents = T,
           FilterCountry = T,
           countryName = c("Portugal"),
           Continent = c("Europe", "Asia"),
           MultipleFetch = T,
           NC = 6) {
    if (MultipleFetch) {
      clu<-makeForkCluster(NC, outfile = paste(NAMEDB,"testing.txt"))
      registerDoParallel(clu)
      
      foreach (i = 1:NC) %dopar% {
        if (i <= NC - 2) {
          if (filter) {
            tryCatch(runs(
              Filter = filter,
              Best = bestMode,
              Fcontinent = FilterContinents,
              Fcountry = FilterCountry,
              countryN = countryName,
              Continent = Continent
            ),error = function(e) {
              cat(conditionMessage(e))
            })
            
          } else{
            tryCatch(runs(
              Filter = F,
              Best = bestMode,
              Best = bestMode,
              Fcontinent = FilterContinents,
              Fcountry = FilterCountry,
              countryN = countryName,
              Continent = Continent
            ),error = function(e) {
              cat(conditionMessage(e))
            })
            
          }
        } else{
          if (FilterSales) {
            tryCatch( runsales(
              Filter = FilterSales,
              Fcontinent = FilterContinents,
              Fcountry = FilterCountry,
              countryN = countryName,
              Continent = Continent
            ),error = function(e) {
              cat(conditionMessage(e))
            })
            
          } else{
            tryCatch(runsales(
              Filter = F,
              Fcontinent = FilterContinents,
              Fcountry = FilterCountry,
              countryN = countryName,
              Continent = Continent
            ),error = function(e) {
              cat(conditionMessage(e))
            })
            
          }
        }
        
        
      }
      stopCluster(clu)
      
    } else{
      datainicial = Sys.Date()
      clu<-makeForkCluster(2, outfile = paste(NAMEDB,"testing.txt"))
      registerDoParallel(clu)
      x = foreach(i = 1:2) %dopar% {
        if (i == 1) {
          if (filter) {
            tryCatch(runs(
              Filter = filter,
              Best = bestMode,
              Fcontinent = FilterContinents,
              Fcountry = FilterCountry,
              countryN = countryName,
              Continent = Continent
            ),error = function(e) {
              cat(conditionMessage(e))
            })
            
          } else{
            tryCatch(runs(
              Filter = F,
              Best = bestMode,
              Fcontinent = FilterContinents,
              Fcountry = FilterCountry,
              countryN = countryName,
              Continent = Continent
            ),error = function(e) {
              cat(conditionMessage(e))
            })
            
          }
        } else{
          if (FilterSales) {
            tryCatch(runsales(
              Filter = FilterSales,
              Fcontinent = FilterContinents,
              Fcountry = FilterCountry,
              countryN = countryName,
              Continent = Continent
            ),error = function(e) {
              cat(conditionMessage(e))
            })
            
          } else{
            tryCatch(runsales(
              Filter = F,
              Fcontinent = FilterContinents,
              Fcountry = FilterCountry,
              countryN = countryName,
              Continent = Continent
            ),error = function(e) {
              cat(conditionMessage(e))
            })
            
          }
        }
        
        
      }
      stopCluster(clu)
      
    }
  }



ModeFecth = "Best" #mode Fetching
modFilterDates = "Week" #filter Hours or Week or Day

checkDates = function(redi, sales) {
  
  redi$date_added_utc = as.POSIXct(redi$date_added_utc, format = "%Y-%m-%dT%H:%M:%SZ")
  sales$date_added_utc = as.POSIXct(sales$date_added_utc, format = "%Y-%m-%dT%H:%M:%SZ")
  date_r_min = min(redi$date_added_utc)
  date_r_max = max(redi$date_added_utc)
  I = which(sales$date_added_utc >= date_r_min &
              sales$date_added_utc <= date_r_max)
  d2 = sales[I, ]
  
  return(d2)
  
  
  
  
}

exportTreatedData=function(x = 1,mod = modFilterDates, modef = ModeFecth){
  library(lubridate)
  
  my_collectionRedi = mongo(collection = STATICREDI, db = NAMEDB)
  my_collectionSales = mongo(collection = STATICSALES , db = NAMEDB)
  print("getting redirects")
  if( redis_file == "" ){ redis_file <- paste(NAMEDB,"Redirects", x , mod, modef,Sys.Date(), sep = "-") }
  redi = my_collectionRedi$export(file(redis_file))
  print("getting sales")
  if( sales_file == "" ){ sales_file <- paste(NAMEDB,"Sales", x , mod, modef,Sys.Date(), sep = "-") }
  sales = my_collectionSales$export(file(sales_file))
  
}



#ajustaFunction() --> esta função vai ajustar o NR e o NS automaticamente a cada x ticks.
ajustaFunctionR<-function(R = NRInitial, idsFromTicksR = 0,  percentIncrease=0.01, thresholdR = 70){
  
  TR<-R 
  
  #ajustar os NR
  #vou ajustar os redirects{
  if((TR/NR)*100>=(100-thresholdR)){
    print(paste("Avoiding loss of threshold of",thresholdR,"of redirects, increasing number of redirects records!", sep = " "))
    assign("NR", NR<-(NR + ceiling(TR*3)),  envir = .GlobalEnv)
    
    
  }else{
    
    
    ToR<-idsFromTicksR
    
    if(ToR < TR){
      
      assign("NR", NR<-(NR + ceiling(NR*percentIncrease)),  envir = .GlobalEnv)
      
    }else if (ToR>= TR){
      
      assign("NR", NR<-(NR - ceiling(NR*percentIncrease)),  envir = .GlobalEnv)
      
      
    }
    
  }
  
  
  
}
ajustaFunctionS<-function( S = NSInitial , idsFromTicksS = 0, percentIncrease=0.01, thresholdS = 70){
  
  
  TS<- S 
  
  
  #ajustar os NS
  
  if((TS/NS)*100>=(100-thresholdS)){
    print(paste("Avoiding loss of threshold of",thresholdS,"of sales, increasing number of sales records!", sep = " "))
    
    assign("NS", NS<-(NS + ceiling(TS*3)),  envir = .GlobalEnv) 
    
    
  }else{ 
    
    ToS<-idsFromTicksS
    
    
    
    
    if(ToS< TS){
      
      
      assign("NS", NS<-(NS + ceiling(NS*percentIncrease)),  envir = .GlobalEnv)
      
    }else if (ToS>= TS){
      
      
      
      assign("NS", NS<-ceiling(NS -  ceiling(NS*percentIncrease)),  envir = .GlobalEnv)
      
      
      
    }
    
  }
}


init() 
RunProgram(filter = T, FilterSales = T, bestMode = "Best", FilterContinents = F, FilterCountry = F, Continent = c("Europe"), MultipleFetch = F)


#example of exporting the data
#if the data collection goes over 1 Day this function will filter to have a full day and creates a new file 
#with the filtred redirects and filter sales within the interval of the redirects (modf is the type of fetching that was made (Best or Test))
exportTreatedData(x = 1 , mod = "Pedro",modef = "Best") 


