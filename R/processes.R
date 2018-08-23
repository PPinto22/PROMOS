# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
library(chron)
library(ggplot2)
library(scales)
library(RColorBrewer)
library(ggstance)

INPUT = '../results/processes2/processes.csv'

dt = data.table(read.csv(INPUT, header=TRUE, sep=','))
dt$time <- as.POSIXct(strptime(dt$time, format="%H:%M:%S"))
dt$eval_time <- as.POSIXct(strptime(dt$eval_time, format="%H:%M:%S"))
dt$ea_time <- as.POSIXct(strptime(dt$ea_time, format="%H:%M:%S"))

dt_med <- dt[, .(time=as.POSIXct(median(time)), eval_time=as.POSIXct(median(eval_time)), ea_time=as.POSIXct(median(ea_time))), by = processes]
dt_melt <- melt(dt_med, measure.vars = c('time', 'eval_time', 'ea_time'), variable.name = 'type', value.name = 'time')
levels(dt_melt$type) <- c('Run', 'Evaluation', 'Evolution')

# https://stackoverflow.com/questions/19235466/how-do-i-plot-time-hhmmss-in-x-axis-in-r
pd <- position_jitterdodge(dodge.width = -1, jitter.height = 5)
ggplot(dt_melt, aes(x=processes, y=time, color=type)) + 
  geom_point(position=pd) + 
  geom_line(position=pd) + 
  scale_x_continuous(breaks=dt_med$processes, minor_breaks = NULL) +
  scale_y_datetime(labels = date_format("%M:%S")) +
  scale_color_brewer(palette = 'Set2') +
  labs(x = 'Processes', y='Time (m:s)', color=NULL) +
  theme_minimal()
