# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# This script reads result files from multiple executions named as, e.g., 'neat_1K(N)_evaluations.csv', where 
# neat_1K is a prefix that identifies the run type, and 'N' is an integer ranging from 1 to the number of runs

# install.packages("hash")
# install.packages("ggpubr")
# install.packages("ggplot2")
# install.packages("data.table")
library(hash)
library(ggplot2)
library(ggpubr)
library(data.table)

# ---- CONFIGURATION ----
RUNS <- 30
RESULTS_DIR <- '../results/NEAT/samples_30runs/'
RUN_TYPES <- c('neat_ALL', 'neat_10K', 'neat_1K', 'neat_100') # These are the prefixes of the result files
RUN_TYPE_LABEL <- hash(keys=RUN_TYPES, values=c('ALL (165K)', '10 000', '1 000', '100'))
SERIES_LABEL <- 'Sample size'
FITNESS_FUNC <- 'AUC'
IMG_OUT_DIR <- 'img/samples_30/'


# ---- SETUP ----
labels_ord = sapply(RUN_TYPES, function(x){RUN_TYPE_LABEL[[x]]})
# Map, for each run type (i.e., each prefix in PREFIX), a list of all respective data files
evals_file_names <- hash()
for(type in RUN_TYPES){
  prefix = paste(RESULTS_DIR, type, sep='')
  if(RUNS > 1)
    prefix = paste(prefix, '(', 1:RUNS, ')', sep='')
  
  evals_file_names[type] <- paste(prefix, '_evaluations.csv', sep='')
}

# Read all data files and store the averages of all runs
evals_dt <- rbindlist(lapply(RUN_TYPES, function(type){
  # Read all runs of the current type
  run_type_dts = lapply(evals_file_names[[type]], function(file_name){
    run_dt = data.table(read.csv(file=file_name, header=TRUE, sep=','))
    # Group by generation
    run_dt = run_dt[ , .(fitness.mean = mean(fitness), fitness.max = max(fitness),
                         neurons.mean = mean(neurons), neurons.max = max(neurons),
                         connections.mean = mean(connections), connections.max = max(connections),
                         time = mean(run_minutes)), by = generation]
  })
  # Join all runs into a single data.table
  run_type_dts = rbindlist(run_type_dts)
  
  # Count how many times each generation occurs
  generation_count = table(run_type_dts$generation)
  # Crop outlier generations that appear in less than 80% of runs
  run_type_dts = run_type_dts[run_type_dts$generation %in% names(generation_count)[generation_count>=0.8*RUNS],]
  
  # Get the average of run_type_dts
  type_avg_dt = run_type_dts[, .(fitness.mean = mean(fitness.mean), fitness.max = mean(fitness.max),
                                 neurons.mean = mean(neurons.mean), neurons.max = mean(neurons.max),
                                 connections.mean = mean(connections.mean),
                                 connections.max = mean(connections.max),
                                 time = mean(time)), by = generation]
  # Add "run_type" column
  run_type = rep(RUN_TYPE_LABEL[[type]], nrow(type_avg_dt))
  type_avg_dt = cbind(type_avg_dt, run_type)
  return(type_avg_dt)
}))

# ---- VISUALIZATION ----
# Create IMG_OUT_DIR
dir.create(file.path(IMG_OUT_DIR), showWarnings = FALSE)

# Plot mean fitness over time
png(filename = paste(IMG_OUT_DIR, 'mean_fitness.png', sep=''))
gg_meanfit <- lapply(labels_ord, function(type){ 
  ggplot(data=subset(evals_dt, run_type==type), aes(time, y=fitness.mean)) + geom_line() +
    labs(x="Run time (min)", y=paste("Mean fitness ", "(", FITNESS_FUNC, ")", sep='')) +
    scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05))
})
do.call(ggarrange, c(gg_meanfit, list(labels=labels_ord)))
dev.off()

# Plot max fitness over time
png(filename = paste(IMG_OUT_DIR, 'max_fitness.png', sep=''))
gg_maxfit <- ggplot(data=evals_dt, aes(time)) + geom_smooth(aes(y=fitness.max, col=run_type), method='loess') +
  labs(x="Run time (min)", y=paste("Max fitness ", "(", FITNESS_FUNC, ")", sep=''), colour="Sample size") +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05))
gg_maxfit
dev.off()

# Plot generations over time
png(filename = paste(IMG_OUT_DIR, 'generations.png', sep=''))
gg_generations <- ggplot(data=evals_dt, aes(time)) + geom_smooth(aes(y=generation, col=run_type), method='loess') +
  labs(x="Run time (min)", y="Generations", colour=SERIES_LABEL)
gg_generations
dev.off()

# Plot complexity (connections) over time
png(filename = paste(IMG_OUT_DIR, 'mean_connections.png', sep=''))
gg_connections <- ggplot(data=evals_dt, aes(time)) + geom_line(aes(y=connections.mean, col=run_type)) +
  labs(x="Run time (min)", y="Mean network connections", colour=SERIES_LABEL)
gg_connections
dev.off()
