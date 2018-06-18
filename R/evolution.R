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

melt_fitness()

# ---- OUTPUTS ----
gsmooth_color <- '#D0D0D0'
# Create OUT_DIR
dir.create(file.path(OUT_DIR), recursive=TRUE, showWarnings=FALSE)

# -- Summary table --
write_summary_table()

# -- Graphs --
if(has_windows){
  # Boxplot of each window's best test result
  png(filename = paste(OUT_DIR, 'window_best_bps.png', sep=''))
  gg_windows = ggplot(data=windows_dt, aes(x=window_factor, y=test_fitness)) +
    geom_boxplot() +
    labs(x="Window", y=fit_label) +
    theme_minimal()
  print(gg_windows)
  dev.off()
  
  # Boxplot average prediction time by discrete generations
  png(filename = paste(OUT_DIR, 'avg_pred_time_by_disc_gen.png', sep=''))
  gg_pred_times <- ggplot(final0_evals_dt, aes(x=generation_factor, y=pred_avg_time)) +
    geom_boxplot() + 
    labs(x='Generation', y='Prediction time (μs)') +
    theme_minimal()
  print(gg_pred_times)
  dev.off()
}

# TODO Fitness scatter plot

# Max and mean train fitness over gens
png(filename = paste(OUT_DIR, 'train_fit_per_gen.png', sep=''))
gg_train_fit <- ggplot(data=evals_fit, aes(x=generation,y=fitness_train, col=mean_or_best)) + 
  geom_smooth(fill=gsmooth_color) + 
  labs(x="Generation", y=fit_label, col='') + 
  scale_color_brewer(palette = 'Set2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
if(has_windows){
  gg_train_fit <- gg_train_fit + geom_vline(xintercept=windows_gen_splits, linetype=3) +
    scale_x_continuous(breaks=windows_avg_dt$generations)
}
print(gg_train_fit)
dev.off()

# Max and mean test fitness over gens
png(filename = paste(OUT_DIR, 'test_fit_per_gen.png', sep=''))
gg_test_fit <- ggplot(data=evals_fit, aes(x=generation,y=fitness_test, col=mean_or_best)) + 
  geom_smooth(fill=gsmooth_color) + 
  labs(x="Generation", y=fit_label, col='') + 
  scale_color_brewer(palette = 'Set2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
if(has_windows){
  gg_test_fit <- gg_test_fit + geom_vline(xintercept=windows_gen_splits, linetype=3) +
    scale_x_continuous(breaks=windows_avg_dt$generations)
}
print(gg_test_fit)
dev.off()

# Max and mean train and test fitness over gens
png(filename = paste(OUT_DIR, 'fits_per_gen.png', sep=''), width = 900, height = 500)
gg_fit <- ggplot(data=evals_fit_long, aes(x=generation,y=fitness, col=mean_or_best)) + 
  geom_smooth(fill=gsmooth_color) + 
  facet_wrap(~train_or_test) +
  labs(x="Generation", y=fit_label, col='') + 
  scale_color_brewer(palette = 'Set2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal() +
  theme(strip.text = element_text(size=12))
if(has_windows){
  gg_fit <- gg_fit + geom_vline(xintercept=windows_gen_splits, linetype=3) +
    scale_x_continuous(breaks=windows_avg_dt$generations)
}
print(gg_fit)
dev.off()

# Network connections over generations
png(filename = paste(OUT_DIR, 'window_connections.png', sep=''))
gg_connections_gen <- ggplot(data=evals_avg_dt, aes(generation)) + 
  geom_smooth(aes(y=connections_mean), method='loess') +
  labs(x="Generation", y="Connections", col='') + 
  theme_minimal()
if(has_windows){
  gg_connections_gen <- gg_connections_gen + geom_vline(xintercept=windows_gen_splits, linetype=3) +
    scale_x_continuous(breaks=windows_avg_dt$generations)
}
print(gg_connections_gen)
dev.off() 

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

# Fitness | Connections
png(filename = paste(OUT_DIR, 'connections_fitness_scatter.png', sep=''))
ggplot(evals_sample_dt, aes(x=connections, y=fitness_test)) +
  geom_point(size=.8, alpha=0.35) +
  labs(x='Connections', y=fit_label) + 
  theme_minimal()
dev.off()
