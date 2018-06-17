# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(ggplot2)
library(ggpubr)
library(data.table)
library(RColorBrewer)

# --- CONFIG
RUNS <- 2
WINDOWS <- 10
RESULTS_DIR <- '../results/2weeks_temp/'
RUN_PREFIX <- 'neat_windows'
OUT_DIR <- 'out/2weeks/'
FITNESS_FUNC <- 'AUC'
DIGITS <- 5

# -- SETUP
source('util.R')
setup()

# Read evaluations
evals_dt <- read_evaluations(evals_file_names)
evals_avg_dt <- group_evals(evals_dt)
evals_sample_dt <- get_evals_sample(evals_dt)

# Read windows
read_windows_or_summaries()
# windows_dt <- read_windows(windows_file_names)
# windows_avg_dt <- group_windows(windows_dt)
# windows_gen_splits <- windows_avg_dt$generations[1:(length(windows_avg_dt$generations)-1)]

# Evaluations at the end of each window
if(has_windows){
  final_evals_dt <- evals_dt[generation %in% windows_avg_dt$generations]
  final0_evals_dt <- rbind(evals_dt[generation==0], final_evals_dt)
  final_evals_dt$generation_factor <- factor(final_evals_dt$generation, ordered = TRUE)
  final0_evals_dt$generation_factor <- factor(final0_evals_dt$generation, ordered = TRUE)
}

# -- WIDE TO LONG CONVERSIONS -- 
if(has_windows){
  # Eval and EA time per window
  window_times <- melt(windows_avg_dt, id.vars = c('window', 'generations'), measure.vars = c('eval_time', 'ea_time'), variable.name = 'state', value.name = 'time')
  levels(window_times$state) <- c('Evaluation', 'Evolution')
}

# Eval times
eval_times <- melt(evals_avg_dt, id.vars=c('generation'), measure.vars=c('eval_time','pred_time','fit_time','build_time'), variable.name='state', value.name='time')
levels(eval_times$state) <- c('Total', 'Predictions', FITNESS_FUNC, 'Build')

# ---- OUTPUTS ----
gsmooth_color <- '#dcdcdc'
# Create OUT_DIR
dir.create(file.path(OUT_DIR), recursive=TRUE, showWarnings=FALSE)

# -- Summary table --
if(!has_windows){
  write.table(summaries_avg_dt, file=paste(OUT_DIR, 'summary.csv', sep=''), row.names = FALSE, sep=',', 
              col.names = c('Time (EA)', 'Time (Evaluation)', 'Time (Total)', 'Generations', 'Fitness (Train)', 'Fitness (Test)', 'Neurons', 'Connections'))
} else{
  write.table(windows_avg_dt[, !"window_factor"], file=paste(OUT_DIR, 'windows.csv', sep=''), row.names = FALSE, sep=',', 
              col.names = c('Window', 'Window Begin', 'Window End', 'Generations', 'Run Time', 'Eval Time', 'EA Time', 'Train Size', 'Train Pos', 'Train Neg',
                            'Test Size', 'Test Pos', 'Test Neg', 'Train Fitness', 'Test Fitness', 'Neurons', 'Connections'))
}

# -- Graphs --
if(has_windows){
  png(filename = paste(OUT_DIR, 'window_best_bps.png', sep=''))
  gg_windows = ggplot(data=windows_dt, aes(x=window_factor, y=test_fitness)) +
    geom_boxplot() +
    labs(x="Window", y=fit_label) +
    theme_minimal()
  print(gg_windows)
  dev.off()
  
  # Max and mean train fitness
  # TODO: Melt; Copy for no window
  png(filename = paste(OUT_DIR, 'window_train_fit.png', sep=''))
  gg_window_train_fit <- ggplot(data=evals_avg_dt, aes(generation)) + 
    geom_smooth(aes(y=fitness_best, col='Best')) +
    geom_smooth(aes(y=fitness_mean, col='Mean')) +
    geom_vline(xintercept=windows_gen_splits, linetype=3) +
    labs(x="Generation", y=fit_label, col='') + 
    scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
    scale_x_continuous(breaks=windows_avg_dt$generations) + 
    theme_minimal()
  print(gg_window_train_fit)
  dev.off()
  
  # Max and mean test fitness
  # TODO: Melt; Copy for no window
  png(filename = paste(OUT_DIR, 'window_test_fit.png', sep=''))
  gg_window_test_fit <- ggplot(data=evals_avg_dt, aes(generation)) + 
    geom_smooth(aes(y=fitness_test_best, col='Max')) +
    geom_smooth(aes(y=fitness_test_mean, col='Mean')) +
    geom_vline(xintercept=windows_gen_splits, linetype=3) +
    labs(x="Generation", y=fit_label, col='') + 
    scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
    scale_x_continuous(breaks=windows_avg_dt$generations) + 
    theme_minimal()
  print(gg_window_test_fit)
  dev.off()
  
  # Network connections
  png(filename = paste(OUT_DIR, 'window_connections.png', sep=''))
  gg_window_connections <- ggplot(data=evals_avg_dt, aes(generation)) + 
    geom_smooth(aes(y=connections_mean), method='loess') +
    geom_vline(xintercept=windows_gen_splits, linetype=3) +
    labs(x="Generation", y="Connections", col='') + 
    theme_minimal()
  print(gg_window_connections)
  dev.off() 
}

# EA vs eval time
png(filename = paste(OUT_DIR, 'ea_eval_time.png', sep=''))
ggplot(window_times, aes(x=generations, y=time, fill=state)) +
  geom_area(position='stack') +
  labs(x='Generation', y='Time', fill='State') + 
  scale_fill_brewer(palette = 'Oranges') +
  theme_minimal()
dev.off()

# Times over generations
png(filename = paste(OUT_DIR, 'times_by_gen.png', sep=''))
ggplot(eval_times, aes(x=generation, y=time, color=state)) +
  geom_smooth(fill=gsmooth_color) +
  labs(x='Generation', y='Time (μs)', color='Times') + 
  scale_color_brewer(palette = 'Set2') +
  theme_minimal()
dev.off()

# Boxplot average prediction time by discrete generations
png(filename = paste(OUT_DIR, 'avg_pred_time_by_disc_gen.png', sep=''))
ggplot(final0_evals_dt, aes(x=generation_factor, y=pred_avg_time)) +
  geom_boxplot() + 
  labs(x='Generation', y='Prediction time (μs)') +
  theme_minimal()
dev.off()

# Fitness | Connections
png(filename = paste(OUT_DIR, 'connections_fitness_scatter.png', sep=''))
ggplot(evals_sample_dt, aes(x=connections, y=fitness_test)) +
  geom_point(size=.8, alpha=0.35) +
  labs(x='Connections', y=fit_label) + 
  theme_minimal()
dev.off()
