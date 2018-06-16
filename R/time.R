# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(ggpubr)
library(data.table)

# --- CONFIG
RUNS <- 16
WINDOWS <- 4
RESULTS_DIR <- '../results/window/'
RUN_PREFIX <- 'neat_windows'
OUT_DIR <- 'out/window2/'
FITNESS_FUNC <- 'AUC'
DIGITS <- 5


# -- SETUP
fit_label <- paste("Fitness ", "(", FITNESS_FUNC, ")", sep='')

prefix = paste(RESULTS_DIR, RUN_PREFIX, sep='')
if(RUNS > 1){prefix = paste(prefix, '(', 1:RUNS, ')', sep='')}
evals_file_names <- paste(prefix, '_evaluations.csv', sep='')

# Read evaluations
evals_dt <- rbindlist(lapply(evals_file_names, function(file_name){
    run_dt = data.table(read.csv(file=file_name, header=TRUE, sep=','))
}))

# Average by generation
evals_avg_dt = evals_dt[ , .(window=round(median(window)), fitness_mean = mean(fitness), fitness_max = max(fitness),
                             fitness_test_mean = mean(fitness_test), fitness_test_best = fitness_test[which.max(fitness)],
                             neurons_mean = mean(neurons), neurons_max = max(neurons), neurons_best = neurons[which.max(fitness)],
                             connections_mean = mean(connections), connections_max = max(connections), connections_best = connections[which.max(fitness)],
                             #build_time=mean(build_time), pred_time=mean(pred_time), pred_avg_time=mean(pred_avg_time),
                             #fit_time=mean(fit_time), eval_time=mean(build_time+pred_time+fit_time),
                             run_time = mean(run_time)), by = generation]

# ---- OUTPUTS ----
# Create OUT_DIR
dir.create(file.path(OUT_DIR), recursive=TRUE, showWarnings=FALSE)

ggplot(evals_avg_dt, aes(x=connections_mean, y=fitness_test_mean)) +
  geom_point(size=1) +
  labs(x='Connections', y=fit_label) + 
  theme_minimal()
  