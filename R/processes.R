# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
library(chron)
library(ggplot2)

INPUT = '../results/processes/processes.csv'

dt = data.table(read.csv(INPUT, header=TRUE, sep=','))
dt$time <- times(dt$time)

dt_med <- dt[, .(time=median(time)), by = processes]

time_labels = as.character(times(dt_med$time))
time_breaks = dt_med[match(unique(time_labels), time_labels)]$time
time_labels = unique(time_labels)

ggplot(dt_med, aes(x=processes, y=time)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous(breaks=dt_med$processes, minor_breaks = NULL) +
  scale_y_continuous(breaks = time_breaks, labels = time_labels, minor_breaks = NULL) +
  labs(x = 'Processes', y='Time') +
  theme_minimal()

