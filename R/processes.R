# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
library(chron)
library(ggplot2)
library(scales)
library(RColorBrewer)
library(ggstance)

INPUT = '../results/processes/processes_new.csv'

dt = data.table(read.csv(INPUT, header=TRUE, sep=','))
# dt$time <- as.POSIXct(strptime(dt$time, format="%H:%M:%S"))
# dt$eval_time <- as.POSIXct(strptime(dt$eval_time, format="%H:%M:%S"))
# dt$ea_time <- as.POSIXct(strptime(dt$ea_time, format="%H:%M:%S"))
dt$time <- dt$time/dt$generations
dt$eval_time <- dt$eval_time/dt$generations
dt$ea_time <- dt$ea_time/dt$generations


# dt_med <- dt[, .(time=as.POSIXct(median(time)), eval_time=as.POSIXct(median(eval_time)), ea_time=as.POSIXct(median(ea_time))), by = processes]
dt_med <- dt[, .(time=median(time), eval_time=(eval_time), ea_time=median(ea_time)), by = processes]
# dt_melt <- melt(dt_med, measure.vars = c('time', 'eval_time', 'ea_time'), variable.name = 'type', value.name = 'time')
# levels(dt_melt$type) <- c('Run', 'Evaluation', 'Evolution')
dt_melt <- melt(dt_med, id.vars=c("processes"), measure.vars = c('eval_time', 'ea_time'), variable.name = 'type', value.name = 'time')
levels(dt_melt$type) <- c('Evaluation', 'Evolution')


# https://stackoverflow.com/questions/19235466/how-do-i-plot-time-hhmmss-in-x-axis-in-r
pd <- position_jitterdodge(dodge.width = -1, jitter.height = 5)
ggplot(dt_melt, aes(x=processes, y=time, color=type)) + 
  # geom_point(position=pd) + 
  # geom_line(position=pd) + 
  geom_point() + geom_line() +
  scale_x_continuous(breaks=dt_med$processes, minor_breaks = NULL) +
  # scale_y_datetime(labels = date_format("%M:%S")) +
  scale_color_brewer(palette = 'Dark2') +
  # labs(x = 'Processes', y='Time (m:s)', color=NULL) +
  labs(x = 'Processes', y='Time (s)', color=NULL) +
  theme_minimal()

ggplot(dt, aes(x=processes, y=eval_time)) + 
  geom_point() + geom_line() +
  scale_x_continuous(breaks=dt_med$processes, minor_breaks = NULL) +
  scale_color_brewer(palette = 'Dark2') +
  labs(x = 'Processes', y='Time (s)', color=NULL) +
  theme_minimal()
