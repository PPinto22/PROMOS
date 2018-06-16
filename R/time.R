# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(ggpubr)
library(data.table)

# --- CONFIG
RUNS <- 16
WINDOWS <- 10
RESULTS_DIR <- '../results/2weeks/'
RUN_PREFIX <- 'neat_windows'
OUT_DIR <- 'out/2weeks/'
FITNESS_FUNC <- 'AUC'
DIGITS <- 5


# -- SETUP
fit_label <- paste("Fitness ", "(", FITNESS_FUNC, ")", sep='')

prefix = paste(RESULTS_DIR, RUN_PREFIX, sep='')
# if(RUNS > 1){prefix = paste(prefix, '(', 1:RUNS, ')', sep='')}
prefix = paste(prefix, '(', c(7, 8), ')', sep='')
evals_file_names <- paste(prefix, '_evaluations.csv', sep='')
windows_file_names <- paste(prefix, '_windows.csv', sep='')

# Read evaluations
evals_dt <- rbindlist(lapply(evals_file_names, function(file_name){
    run_dt = data.table(read.csv(file=file_name, header=TRUE, sep=','))
    run_dt$eval_time = run_dt$build_time + run_dt$pred_time + run_dt$fit_time
    run_dt
}))
# Average by generation
evals_avg_dt = evals_dt[ , .(window=round(median(window)), fitness_mean = mean(fitness), fitness_max = max(fitness),
                             fitness_test_mean = mean(fitness_test), fitness_test_best = fitness_test[which.max(fitness)],
                             neurons_mean = mean(neurons), neurons_max = max(neurons), neurons_best = neurons[which.max(fitness)],
                             connections_mean = mean(connections), connections_max = max(connections), connections_best = connections[which.max(fitness)],
                             build_time=mean(build_time), pred_time=mean(pred_time), pred_avg_time=mean(pred_avg_time),
                             fit_time=mean(fit_time), eval_time=mean(eval_time),
                             run_time = mean(run_time)), by = generation]

# Read windows
windows_dt <- rbindlist(lapply(windows_file_names, function(file_name){
  summary_dt = data.table(read.csv(file=file_name, header=TRUE, sep=','))
}))
windows_dt$window <- factor(windows_dt$window, ordered = TRUE)

windows_avg_dt <- windows_dt[, .(begin_date=first(begin_date), end_date=first(end_date), generations=mean(generations), run_minutes=mean(run_minutes),
                                 train_size=mean(train_size), train_positives=mean(train_positives), train_negatives=mean(train_negatives),
                                 test_size=mean(test_size), test_positives=mean(test_positives), test_negatives=mean(test_negatives),
                                 train_fitness=mean(train_fitness), test_fitness=mean(test_fitness), 
                                 best_neurons=round(mean(best_neurons)), best_connections=round(mean(best_connections))), by = window]
windows_gen_splits <- windows_avg_dt$generations[1:(length(windows_avg_dt$generations)-1)]

final_evals_dt <- evals_dt[generation %in% windows_avg_dt$generations]
final0_evals_dt <- rbind(evals_dt[generation==0], final_evals_dt)
final_evals_dt$generation <- factor(final_evals_dt$generation, ordered = TRUE)
final0_evals_dt$generation <- factor(final0_evals_dt$generation, ordered = TRUE)

# ---- OUTPUTS ----
# Create OUT_DIR
dir.create(file.path(OUT_DIR), recursive=TRUE, showWarnings=FALSE)

gsmooth_color <- '#dcdcdc'

# Times over generations
png(filename = paste(OUT_DIR, 'times_by_gen.png', sep=''))
ggplot(evals_avg_dt, aes(x=generation)) +
  geom_smooth(aes(y=build_time, color='Build'), fill=gsmooth_color) +
  geom_smooth(aes(y=pred_time, color='Predictions'), fill=gsmooth_color) +
  geom_smooth(method=lm, aes(y=fit_time, color='AUC'), fill=gsmooth_color) +
  geom_smooth(aes(y=eval_time, color='Total'), fill=gsmooth_color) +
  labs(x='Generation', y='Time (μs)', color='Times') + 
  theme_minimal()
dev.off()
  
# Boxplot average prediction time by discrete generations
png(filename = paste(OUT_DIR, 'avg_pred_time_by_disc_gen.png', sep=''))
ggplot(final0_evals_dt, aes(x=generation)) +
  geom_boxplot(aes(y = pred_avg_time)) + 
  labs(x='Generation', y='Prediction time (μs)') +
  theme_minimal()
dev.off()

# Fitness | Connections
ggplot(evals_avg_dt, aes(x=connections_mean, y=fitness_test_mean)) +
  geom_point(size=1) +
  labs(x='Connections', y=fit_label) + 
  theme_minimal()