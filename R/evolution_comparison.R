# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# This script reads result files from multiple executions named as, e.g., 'neat_1K(N)_evaluations.csv', where 
# neat_1K is a prefix that identifies the run type, and 'N' is an integer ranging from 1 to the number of runs

# ---- SETUP ----
source('config.R')
source('util.R')
setup(multi_types=TRUE)

# Read evaluations
evals_avg_dt <- rbindlist(lapply(1:n_run_types, function(i){
  run_type_dt = group_evals(read_evaluations(evals_file_names[[i]]), crop=0.2)  # Read all evals; average each run individually; then average the average of every run
  run_type_dt$run_type = rep(RUN_TYPE_LABELS[i], nrow(run_type_dt))   # Add "run_type" column
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
sink(paste(OUT_DIR, 'ttest.txt', sep = ''))
if(!has_windows){
  pairwise.t.test(summaries_dt$test_fit, summaries_dt$run_type)
} else{
  pairwise.t.test(windows_dt$test_fitness, windows_dt$run_type, paired=TRUE)
}
sink()

# -- Graphs --
if(!has_windows){
  # Boxplot best test fitness by run type
  png(filename = paste(OUT_DIR, 'fitness_best_bp.png', sep=''))
  gg_best_testfit <-  ggplot(data=summaries_dt, aes(x=run_type, y=test_fit)) +
    geom_boxplot() +
    geom_shadowtext(data = summaries_avg_dt, aes(x=run_type, y=test_fit, label=sprintf("%.4f", round(test_fit, digits = 4))), size=5) +
    labs(x=SERIES_LABEL, y=FITNESS_FUNC) + 
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
  
  # Generations bar plot
  png(filename = paste(OUT_DIR, 'generations_bar.png', sep=''))
  gg_generations_bar <- ggplot(data=summaries_avg_dt, aes(x=run_type, y=generations)) +
    geom_bar(stat='identity') + 
    geom_text(aes(label=generations), size = 5, nudge_y=100) +
    labs(x=SERIES_LABEL, y='Generations') +
    theme_minimal()
  print(gg_generations_bar)
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

# best train fitness over time
png(filename = paste(OUT_DIR, 'best_train_fit_by_time.png', sep=''))
gg_best_train_fit_time <- ggplot(data=evals_avg_dt, aes(run_time)) + 
  geom_smooth(aes(y=fitness_best, col=run_type), fill=gsmooth_fill) +
  labs(x="Run time (min)", y=fit_label, col=SERIES_LABEL) +
  scale_color_brewer(palette = 'Set2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
print(gg_best_train_fit_time)
dev.off()

# best train fitness over gens
png(filename = paste(OUT_DIR, 'best_train_fit_by_gens.png', sep=''))
gg_best_train_fit_gens <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_smooth(aes(y=fitness_best, col=run_type), fill=gsmooth_fill) +
  labs(x="Generation", y=fit_label, col=SERIES_LABEL) +
  scale_color_brewer(palette = 'Set2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
print(gg_best_train_fit_gens)
dev.off()

# best test fitness over gens
png(filename = paste(OUT_DIR, 'best_test_fit_by_gens.png', sep=''))
gg_best_test_fit_gens <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_smooth(aes(y=fitness_test_best, col=run_type), fill=gsmooth_fill) +
  labs(x="Generation", y=FITNESS_FUNC, col=SERIES_LABEL) +
  scale_color_brewer(palette = 'Set2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
print(gg_best_test_fit_gens)
dev.off()

# generations over time
png(filename = paste(OUT_DIR, 'generations_by_time.png', sep=''))
gg_generations_time <- ggplot(data=evals_avg_dt, aes(run_time)) + 
  geom_smooth(aes(y=generation, col=run_type), fill=gsmooth_fill) +
  labs(x="Run time (min)", y="Generations", col=SERIES_LABEL) + 
  scale_color_brewer(palette = 'Set2') +
  theme_minimal()
print(gg_generations_time)
dev.off()

# mean complexity (connections) over generations
png(filename = paste(OUT_DIR, 'connections_mean_by_gens.png', sep=''))
gg_mean_connections_gen <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_line(aes(y=connections_mean, col=run_type)) +
  labs(x="Generation", y="Connections", col=SERIES_LABEL) + 
  scale_color_brewer(palette = 'Set2') +
  theme_minimal()
gg_mean_connections_gen <- add_window_vlines(gg_mean_connections_gen)
print(gg_mean_connections_gen)
dev.off()

# mean complexity (neurons) over generations
png(filename = paste(OUT_DIR, 'neurons_mean_by_gens.png', sep=''))
gg_mean_neurons_gen <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_line(aes(y=neurons_mean, col=run_type)) +
  labs(x="Generation", y="Neurons", col=SERIES_LABEL) + 
  scale_color_brewer(palette = 'Set2') +
  theme_minimal()
gg_mean_neurons_gen <- add_window_vlines(gg_mean_neurons_gen)
print(gg_mean_neurons_gen)
dev.off()
