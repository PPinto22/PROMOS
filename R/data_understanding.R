# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(jsonlite)
library(data.table)
source('util.R')

# Config
INPUT <- '../data/2weeks/ColetasPROMOSBESTTreated2018-06-13.json'
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

prop_dt = data.table(prop.table(sort(table(dt$idoperator), decreasing = T)))
prop_dt$ID <- seq.int(nrow(prop_dt))
# ggplot(prop_dt, aes(x=reorder(V1, -N), y=N)) + geom_bar(stat='identity')

cut = first(which(cumsum(prop_dt$N) >= 0.95))
freq_plot <- ggplot(prop_dt, aes(x=ID, y=N*100)) + 
  geom_bar(stat='identity', lwd=.1, colour='black') + 
  geom_vline(xintercept = cut, color='red') +
  labs(x = 'Rank', y='Relative frequency (%)') +
  theme_minimal()
freq_plot <- freq_plot +
  scale_x_continuous(breaks = sort(c(ggplot_build(freq_plot)$layout$panel_ranges[[1]]$x.major_source, cut)), minor_breaks = F)
print(freq_plot)


sapply(dt, function(x){length(levels(as.factor(x)))})

levels(as.factor(dt$accmanager))
