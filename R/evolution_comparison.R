# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# This script reads result files from multiple executions named as, e.g., 'neat_1K(N)_evaluations.csv', where 
# neat_1K is a prefix that identifies the run type, and 'N' is an integer ranging from 1 to the number of runs

# ---- SETUP ----
source('config.R')
source('util.R')
setup(multi_types=TRUE)

# Read evaluations
evals_avg_dt <- rbindlist(lapply(1:n_run_types, function(i){
  run_type_dt = group_evals(read_evaluations(evals_file_names[[i]], windows=WINDOWS[[i]]), crop=0.2)  # Read all evals; average each run individually; then average the average of every run
  run_type_dt$run_type = rep(RUN_TYPE_LABELS[i], nrow(run_type_dt))   # Add "run_type" column
  return(run_type_dt)
}))
evals_avg_dt$run_type <- factor(evals_avg_dt$run_type, levels = RUN_TYPE_LABELS)


# Read generations
gens_avg_dt <- rbindlist(lapply(1:n_run_types, function(i){
  run_type_dt = group_gens(read_generations(gens_file_names[[i]], windows=WINDOWS[[i]]))
  run_type_dt$run_type = rep(RUN_TYPE_LABELS[i], nrow(run_type_dt))
  return(run_type_dt)
}))
gens_avg_dt$run_type <- factor(gens_avg_dt$run_type, levels = RUN_TYPE_LABELS)

set_gen_breaks(n=5)

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
  gg_best_testfit <- ggplot(data=summaries_dt, aes(x=run_type, y=test_fit)) +
    geom_boxplot(fill=gsmooth_fill) +
    geom_shadowtext(data = summaries_avg_dt, aes(x=run_type, y=test_fit, label=sprintf("%.4f", round(test_fit, digits = 4))), size=6) +
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
  # Box plot of last windows results
  png(filename = paste(OUT_DIR, 'window_final_best_bp.png', sep=''))
  gg_windows_final = ggplot(data=windows_final_dt, aes(x=run_type, y=test_fitness)) +
    geom_boxplot() +
    labs(x=SERIES_LABEL, y=FITNESS_FUNC) +
    theme_minimal()
  print(gg_windows_final)
  dev.off()
  
  # Box plot of each window best, split by run_type
  png(filename = paste(OUT_DIR, 'window_best_bps.png', sep=''))
  gg_windows = ggplot(data=windows_dt, aes(x=window_factor, y=test_fitness)) +
    geom_boxplot() +
    facet_wrap(~run_type) +
    labs(x="Window", y=fit_label) +
    theme_minimal()
  print(gg_windows)
  dev.off()
  
  # Line plot
  windows_avg_dt$mode <- factor(sapply(windows_avg_dt$run_type, function(x){ strsplit(as.character(x), " ")[[1]][1] }), levels=c("BEST", "TEST"))
  windows_avg_dt$algorithm <- factor(sapply(windows_avg_dt$run_type, function(x){ strsplit(as.character(x), " ")[[1]][2] }), levels=c("NEAT", "NEATP", "HyperNEAT"))
  windows_avg_dt$encoding <- factor(sapply(windows_avg_dt$run_type, function(x){ strsplit(as.character(x), " ")[[1]][3] }), levels=c("RAW", "IDF"))
  gg_windows_line = ggplot(data = windows_avg_dt, aes(x=window+1, y=test_fitness, col=algorithm)) +
    facet_grid(rows=vars(mode), cols=vars(encoding)) +
    geom_point() +
    geom_line() +
    labs(x='Window', y='AUC', col='') +
    scale_x_continuous(breaks=seq(1,10), minor_breaks = NULL) +
    scale_y_continuous(limits=c(0.675, 0.825), breaks=seq(0.6,0.825,0.025)) +
    scale_color_brewer(palette = 'Dark2') +
    theme_minimal() +
    theme(legend.position="top")
  print(gg_windows_line)
}

# best train fitness over time
png(filename = paste(OUT_DIR, 'best_train_fit_by_time.png', sep=''))
gg_best_train_fit_time <- ggplot(data=evals_avg_dt, aes(run_time)) + 
  geom_smooth(aes(y=fitness_best, col=run_type), fill=gsmooth_fill) +
  labs(x="Run time (min)", y=fit_label, col=SERIES_LABEL) +
  scale_color_brewer(palette = 'Dark2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
print(gg_best_train_fit_time)
dev.off()

# best train fitness over gens
png(filename = paste(OUT_DIR, 'best_train_fit_by_gens.png', sep=''))
gg_best_train_fit_gens <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_smooth(aes(y=fitness_best, col=run_type), fill=gsmooth_fill) +
  labs(x="Generation", y=fit_label, col=SERIES_LABEL) +
  scale_color_brewer(palette = 'Dark2') +
  # scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
print(gg_best_train_fit_gens)
dev.off()

# best test fitness over gens
png(filename = paste(OUT_DIR, 'best_test_fit_by_gens.png', sep=''))
gg_best_test_fit_gens <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_smooth(aes(y=fitness_test_best, col=run_type), fill=gsmooth_fill, method='loess', span=0.1) +
  labs(x="Generation", y=FITNESS_FUNC, col=SERIES_LABEL) +
  scale_color_brewer(palette = 'Dark2') +
  # scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) +
  scale_x_continuous(breaks=gen_breaks) +
  theme_minimal() + 
  theme(legend.position="top")
print(gg_best_test_fit_gens)
dev.off()

# Best test fitness over gens per algorithm/encoding (AH HOC)
evals_avg_dt$mode <- factor(sapply(evals_avg_dt$run_type, function(x){ strsplit(as.character(x), " ")[[1]][1] }), levels=c("BEST", "TEST"))
evals_avg_dt$algorithm <- factor(sapply(evals_avg_dt$run_type, function(x){ strsplit(as.character(x), " ")[[1]][2] }), levels=c("NEAT", "NEATP", "HyperNEAT"))
evals_avg_dt$encoding <- factor(sapply(evals_avg_dt$run_type, function(x){ strsplit(as.character(x), " ")[[1]][3] }), levels=c("RAW", "IDF"))
gg_best_test_fit_comp <- ggplot(data=evals_avg_dt, aes(generation, y=fitness_best, col=algorithm)) + 
  facet_grid(rows=vars(mode), cols=vars(encoding)) +
  geom_smooth(fill=gsmooth_fill, method='loess', span=0.001) +
  labs(x="Generation", y=FITNESS_FUNC, col=SERIES_LABEL) +
  scale_color_brewer(palette = 'Dark2') +
  scale_y_continuous(limits=c(0.65, 0.8), breaks=seq(0.6,0.8,0.025)) +
  scale_x_continuous(breaks=gen_breaks, labels = human_numbers) +
  theme_minimal() +
  theme(legend.position="top")
print(gg_best_test_fit_comp)

# best test fitness over gens (zoomed)
png(filename = paste(OUT_DIR, 'best_test_fit_by_gens_zoom.png', sep=''))
gg_best_test_fit_gens_zoom <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_smooth(aes(y=fitness_test_best, col=run_type), fill=gsmooth_fill) +
  labs(x="Generation", y=FITNESS_FUNC, col=SERIES_LABEL) +
  scale_color_brewer(palette = 'Dark2') +
  # scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  scale_x_continuous(breaks=gen_breaks) +
  theme_minimal()
print(gg_best_test_fit_gens_zoom)
dev.off()

# generations over time
png(filename = paste(OUT_DIR, 'generations_by_time.png', sep=''))
gg_generations_time <- ggplot(data=evals_avg_dt, aes(run_time)) + 
  geom_smooth(aes(y=generation, col=run_type), fill=gsmooth_fill) +
  labs(x="Run time (min)", y="Generations", col=SERIES_LABEL) + 
  scale_color_brewer(palette = 'Dark2') +
  theme_minimal()
print(gg_generations_time)
dev.off()

# times by generation
png(filename = paste(OUT_DIR, 'times_by_gen.png', sep=''))
gg_times_by_gen <- ggplot(data=gens_avg_dt, aes(generation)) +
  facet_wrap(~run_type) +
  geom_line(aes(y=ea_time, col='ea_time')) +
  geom_line(aes(y=eval_time, col='eval_time')) +
  scale_color_brewer(palette = 'Dark2') +
  theme_minimal()
print(gg_times_by_gen)
dev.off()

# mean complexity (connections) over generations
png(filename = paste(OUT_DIR, 'connections_mean_by_gens.png', sep=''))
gg_mean_connections_gen <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_line(aes(y=connections_mean, col=run_type)) +
  labs(x="Generation", y="Connections", col=SERIES_LABEL) + 
  scale_color_brewer(palette = 'Dark2') +
  scale_x_continuous(breaks=gen_breaks) +
  theme_minimal() + 
  theme(legend.position="top")
gg_mean_connections_gen <- add_window_vlines(gg_mean_connections_gen)
print(gg_mean_connections_gen)
dev.off()

# prediction time over gens
gg_pred_time_gen <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_smooth(aes(y=pred_avg_time, col=run_type), fill=gsmooth_fill) +
  labs(x="Generation", y="Prediction time (µs)", col=SERIES_LABEL) + 
  scale_color_brewer(palette = 'Dark2') +
  scale_x_continuous(breaks=gen_breaks) +
  theme_minimal()
gg_pred_time_gen <- add_window_vlines(gg_pred_time_gen)
print(gg_pred_time_gen)

# mean complexity (connections) over time
png(filename = paste(OUT_DIR, 'connections_mean_by_time.png', sep=''))
gg_mean_connections_time <- ggplot(data=evals_avg_dt, aes(run_time)) + 
  geom_line(aes(y=connections_mean, col=run_type)) +
  labs(x="Run time (minutes)", y="Connections", col=SERIES_LABEL) + 
  scale_color_brewer(palette = 'Dark2') +
  theme_minimal()
gg_mean_connections_time <- add_window_vlines(gg_mean_connections_gen)
print(gg_mean_connections_time)
dev.off()

# mean complexity (neurons) over generations
png(filename = paste(OUT_DIR, 'neurons_mean_by_gens.png', sep=''))
gg_mean_neurons_gen <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_line(aes(y=neurons_mean, col=run_type)) +
  labs(x="Generation", y="Neurons", col=SERIES_LABEL) + 
  scale_color_brewer(palette = 'Dark2') +
  theme_minimal()
gg_mean_neurons_gen <- add_window_vlines(gg_mean_neurons_gen)
print(gg_mean_neurons_gen)
dev.off()
