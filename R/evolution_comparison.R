# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# This script reads result files from multiple executions named as, e.g., 'neat_1K(N)_evaluations.csv', where 
# neat_1K is a prefix that identifies the run type, and 'N' is an integer ranging from 1 to the number of runs

library(hash)
library(ggplot2)
library(ggpubr)
library(data.table)
library(chron)

# ---- CONFIGURATION ----
RUNS <- 2
WINDOWS <- 10
RESULTS_DIR <- '../results/2weeks_temp/' # These are the prefixes of the result files
RUN_TYPES <- c('neat_windows')
RUN_TYPE_LABEL <- hash(keys=RUN_TYPES, values=c('Sliding window'))
SERIES_LABEL <- 'Run'
OUT_DIR <- 'out/2weeks/'

FITNESS_FUNC <- 'AUC'
DIGITS <- 5

# if(length(RUN_TYPES) <= 1){stop("Multiple run types are required for a comparison")}
# ---- SETUP ----
source('util.R')
setup(multi_types=TRUE)

# Read evaluations
evals_avg_dt <- rbindlist(lapply(RUN_TYPES, function(type){
  run_type_dt = group_evals(read_evaluations(evals_file_names[[type]]))  # Read all evals; average each run individually; then average the average of every run
  run_type_dt$run_type = rep(RUN_TYPE_LABEL[[type]], nrow(run_type_dt))  # Add "run_type" column
  return(run_type_dt)
}))

# Read windows/summaries
read_windows_or_summaries()

# -- WIDE TO LONG CONVERSIONS -- 
melt_fitness()

# ---- OUTPUTS ----
# Create OUT_DIR
dir.create(file.path(OUT_DIR), recursive=TRUE, showWarnings=FALSE)

# -- Summary table --
write_summary_table()

# -- Statistical tests --
if(!has_windows){
  sink(paste(OUT_DIR, 'ttest.txt', sep = ''))
  pairwise.t.test(summaries_dt$train_fit, summaries_dt$run_type,  paired=TRUE)
  sink()
}

# -- Graphs --
if(!has_windows){
  # Boxplot best test fitness by run type
  png(filename = paste(OUT_DIR, 'fitness_best_bp.png', sep=''))
  gg_best_testfit <- ggplot(data=summaries_dt, aes(x=run_type, y=test_fit)) +
    geom_boxplot() +
    labs(x=SERIES_LABEL, y=fit_label) + 
    theme_minimal()
  gg_best_testfit
  dev.off()
  
  # Boxplot #connections of the best individual by run type
  png(filename = paste(OUT_DIR, 'connections_best_bp.png', sep=''))
  gg_best_connections <- ggplot(data=summaries_dt, aes(x=run_type, y=connections)) +
    geom_boxplot() +
    labs(x=SERIES_LABEL, y="Connections") +
    theme_minimal()
  gg_best_connections
  dev.off()
} else{
  # Box plot of each window best, split by run_type
  png(filename = paste(OUT_DIR, 'window_best_bps.png', sep=''))
  gg_windows = ggplot(data=windows_dt, aes(x=window_factor, y=test_fitness)) +
    geom_boxplot() +
    facet_wrap(~run_type) +
    labs(x="Window", y=fit_label) +
    theme_minimal()
  print(gg_windows)
  dev.off()
}

# TODO:
# Fitness plot - best test fitness; color(run_tyoe)
# Fitness plot - color(train best/mean), split(run_type)

# Plot mean fitness over time
png(filename = paste(OUT_DIR, 'fitness_mean.png', sep=''))
gg_meanfit <- ggplot(data=evals_dt, aes(run_time)) + 
  geom_smooth(aes(y=fitness_mean, col=run_type), fill=gsmooth_fill) +
  labs(x="Run time (min)", y=fit_label, col=SERIES_LABEL) +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) +
  theme_minimal()
gg_meanfit
dev.off()

# Plot max fitness over time
png(filename = paste(OUT_DIR, 'fitness_best.png', sep=''))
gg_maxfit <- ggplot(data=evals_dt, aes(run_time)) + 
  geom_smooth(aes(y=fitness_best, col=run_type)) +
  labs(x="Run time (min)", y=fit_label, col="Sample size") +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
gg_maxfit
dev.off()

# Plot generations over time
png(filename = paste(OUT_DIR, 'generations.png', sep=''))
gg_generations <- ggplot(data=evals_dt, aes(run_time)) + 
  geom_smooth(aes(y=generation, col=run_type), fill=gsmooth_fill) +
  labs(x="Run time (min)", y="Generations", col=SERIES_LABEL) + 
  theme_minimal()
gg_generations
dev.off()

# Plot mean complexity (connections) over time
png(filename = paste(OUT_DIR, 'connections_mean.png', sep=''))
gg_connections <- ggplot(data=evals_dt, aes(run_time)) + 
  geom_line(aes(y=connections_mean, col=run_type)) +
  labs(x="Run time (min)", y="Connections", col=SERIES_LABEL) + 
  theme_minimal()
gg_connections
dev.off()

# Plot complexity (connections) of the best individual over time
png(filename = paste(OUT_DIR, 'connections_best.png', sep=''))
gg_connections <- ggplot(data=evals_dt, aes(run_time)) + 
  geom_line(aes(y=connections_best, col=run_type)) +
  labs(x="Run time (min)", y="Connections", col=SERIES_LABEL) + 
  theme_minimal()
gg_connections
dev.off()

# Plot mean complexity (connections) over generations
png(filename = paste(OUT_DIR, 'connections_mean_by_gens.png', sep=''))
gg_mean_connections_gen <- ggplot(data=evals_dt, aes(generation)) + 
  geom_line(aes(y=connections_mean, col=run_type)) +
  labs(x="Generation", y="Connections", col=SERIES_LABEL) + 
  theme_minimal()
gg_mean_connections_gen
dev.off()

# Plot complexity (connections) of the best individual over generations
png(filename = paste(OUT_DIR, 'connections_best_by_gens.png', sep=''))
gg_best_connections_gen <- ggplot(data=evals_dt, aes(generation)) + 
  geom_line(aes(y=connections_best, col=run_type)) +
  labs(x="Generation", y="Connections", col=SERIES_LABEL) + 
  theme_minimal()
gg_best_connections_gen
dev.off()
