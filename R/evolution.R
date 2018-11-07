# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# -- SETUP
source('config.R')
source('util.R')
setup()

# Read generations
gens_dt <- read_generations(gens_file_names, windows=WINDOWS)
gens_avg_dt <- group_gens(gens_dt)
gens_single_dt <- gens_dt[run==1]

# Read evaluations
evals_dt <- read_evaluations(evals_file_names, windows=WINDOWS)
evals_avg_dt <- group_evals(evals_dt)
evals_single_dt <- group_evals(evals_dt[run==1], crop=FALSE)
evals_sample_dt <- get_evals_sample(evals_dt)
evals_pairs <- evals_sample_dt[,colnames(evals_sample_dt) %in% c('connections','eval_time', 'fitness'), with=FALSE]
setcolorder(evals_pairs, c('connections', 'eval_time', 'fitness'))
set_gen_breaks(n=5)
# Read windows
read_windows_or_summaries()

# All evaluations of some generations
if(has_windows){
  disc_evals_dt <- evals_dt[ generation==0 | (generation+1) %in% windows_avg_dt$generations]
} else{
  gens = max(evals_dt$generation)+1
  disc_evals_dt <- evals_dt[ generation==0 | (generation+1) %in% seq(gens%/%10, gens, gens%/%10)]
}
disc_evals_dt$generation <- disc_evals_dt$generation+1
disc_evals_dt$generation_factor <- factor(disc_evals_dt$generation, ordered = TRUE)

# -- WIDE TO LONG CONVERSIONS --
if(has_windows){
  # Eval and EA time per window
  window_times <- melt(windows_avg_dt, id.vars = c('window', 'generations'), measure.vars = c('eval_time', 'ea_time'), variable.name = 'state', value.name = 'time')
  levels(window_times$state) <- c('Evaluation', 'Evolution')
}

# Eval and EA time per gen
gen_times <- melt(gens_avg_dt, id.vars = c('generation'), measure.vars = c('eval_time', 'ea_time'), variable.name = 'state', value.name = 'time')
levels(gen_times$state) <- c('Evaluation', 'Evolution')

# Eval times
eval_times <- melt(evals_avg_dt, id.vars=c('generation'), measure.vars=c('eval_time','pred_time','fit_time','build_time'), variable.name='state', value.name='time')
levels(eval_times$state) <- c('Total', 'Predictions', FITNESS_FUNC, 'Build')

melt_fitness()

# Connection, eval_time and fitness deviation
evals_dev <- evals_avg_dt[, c('window','generation', 'connections_mean', 'fitness_mean', 'eval_time')]
for(col in c('connections_mean', 'fitness_mean', 'eval_time')){
  evals_dev[[col]] <- deviation(evals_dev[[col]])
}
evals_dev <- melt(evals_dev, measure.vars=c('connections_mean', 'eval_time', 'fitness_mean'))
levels(evals_dev$variable) <- c('Connections', 'Eval time', 'Fitness')

# Mutation probabilities
if(!is.null(gens_single_dt)){
  mutations <- melt(gens_single_dt, measure.vars=c('add_neuron', 'rem_neuron', 'add_link', 'rem_link'), variable.name='mutation', value.name='prob')
  levels(mutations$mutation) <- c('Add neuron', 'Rem neuron', 'Add link', 'Rem link')
}

# ---- OUTPUTS ----
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
  
  # Test AUC per window (line)
  png(filename = paste(OUT_DIR, 'test_per_window.png', sep=''))
  gg_windows_line = ggplot(data=windows_avg_dt, aes(x=window, y=test_fitness)) +
    geom_line() +
    geom_point() +
    scale_x_continuous(breaks = seq(1, max(windows_avg_dt$window)+1)) +
    labs(x="Window", y=fit_label) +
    theme_minimal()
  print(gg_windows_line)
  dev.off()

  # EA vs eval time (per window)
  png(filename = paste(OUT_DIR, 'windows_ea_eval_time.png', sep=''))
  gg_window_ea_eval <- ggplot(window_times, aes(x=generations, y=time, fill=state)) +
    geom_area(position='stack') +
    labs(x='Generation', y='Time (min)', fill='State') +
    scale_fill_brewer(palette = 'Oranges') +
    geom_vline(xintercept=windows_gen_splits, linetype=2, size=0.2) +
    scale_x_continuous(breaks=windows_avg_dt$generations, minor_breaks = NULL) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
  print(gg_window_ea_eval)
  dev.off()
}

# EA vs eval time (per gen)
png(filename = paste(OUT_DIR, 'gens_ea_eval_time.png', sep=''))
gg_gens_ea_eval <- ggplot(gen_times, aes(x=generation, y=time, fill=state)) +
  geom_area(position='stack') +
  labs(x='Generation', y='Time', fill='State') +
  scale_fill_brewer(palette = 'Oranges') +
  theme_minimal()
print(gg_gens_ea_eval)
dev.off()

# EA vs eval time (per gen, smooth)
png(filename = paste(OUT_DIR, 'gens_ea_eval_time_smooth.png', sep=''))
gg_gens_ea_eval_smooth <- ggplot(gen_times, aes(x=generation, y=time, fill=state)) +
  stat_smooth(geom='area', position='stack') +
  labs(x='Generation', y='Time', fill='State') +
  scale_fill_brewer(palette = 'Oranges') +
  theme_minimal()
print(gg_gens_ea_eval_smooth)
dev.off()

# Boxplot average prediction time by discrete generations
png(filename = paste(OUT_DIR, 'avg_pred_time_by_disc_gen.png', sep=''))
gg_pred_times <- ggplot(disc_evals_dt, aes(x=generation_factor, y=pred_avg_time)) +
  geom_boxplot() +
  labs(x='Generation', y='Prediction time (μs)') +
  theme_minimal()
print(gg_pred_times)
dev.off()

# Boxplot connections by discrete generations
png(filename = paste(OUT_DIR, 'connections_by_disc_gen.png', sep=''))
  gg_disc_connections <- ggplot(disc_evals_dt, aes(x=generation_factor, y=connections)) +
    geom_boxplot() +
    labs(x='Generation', y='Connections') +
    theme_minimal()
  print(gg_disc_connections)
dev.off()

# Max and mean train fitness over gens
png(filename = paste(OUT_DIR, 'train_fit_per_gen.png', sep=''))
gg_train_fit <- ggplot(data=evals_fit, aes(x=generation,y=fitness_train, col=mean_or_best)) +
  geom_smooth(fill=gsmooth_fill) +
  labs(x="Generation", y=fit_label, col='') +
  scale_color_brewer(palette = 'Dark2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) +
  theme_minimal()
gg_train_fit <- add_window_vlines(gg_train_fit)
print(gg_train_fit)
dev.off()

# Max and mean train fitness over gens (line)
png(filename = paste(OUT_DIR, 'train_fit_line_per_gen.png', sep=''))
gg_train_fit_line <- ggplot(data=evals_fit, aes(x=generation,y=fitness_train, col=mean_or_best)) +
  geom_line() +
  labs(x="Generation", y=fit_label, col='') +
  scale_color_brewer(palette = 'Dark2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) +
  theme_minimal()
gg_train_fit_line <- add_window_vlines(gg_train_fit_line)
print(gg_train_fit_line)
dev.off()

# Max and mean test fitness over gens
png(filename = paste(OUT_DIR, 'test_fit_per_gen.png', sep=''))
evals_fit_sample <- evals_fit[sample(1:nrow(evals_fit), 200000)]
gg_test_fit <- ggplot(data=evals_fit_sample, aes(x=generation,y=fitness_test, col=mean_or_best)) +
  geom_smooth(fill=gsmooth_fill, span=0.01, method='loess') +
  labs(x="Generation", y=FITNESS_FUNC, col='') +
  scale_color_brewer(palette = 'Dark2') +
  # scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
gg_test_fit <- add_window_vlines(gg_test_fit, labels = human_numbers, scale_break_freq = 3)
print(gg_test_fit)
dev.off()

# Max and mean test fitness over gens (line)
png(filename = paste(OUT_DIR, 'test_fit_line_per_gen.png', sep=''))
gg_test_fit_line <- ggplot(data=evals_fit, aes(x=generation,y=fitness_test, col=mean_or_best)) +
  geom_line() +
  labs(x="Generation", y=FITNESS_FUNC, col='') +
  scale_color_brewer(palette = 'Dark2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
gg_test_fit_line <- add_window_vlines(gg_test_fit_line, labels = human_numbers)
print(gg_test_fit_line)
dev.off()

# Max and mean train and test fitness over gens
png(filename = paste(OUT_DIR, 'fits_per_gen.png', sep=''), width = 900, height = 500, res=100)
gg_fit <- ggplot(data=evals_fit_long, aes(x=generation,y=fitness, col=mean_or_best)) +
  geom_smooth(fill=gsmooth_fill) +
  facet_wrap(~train_or_test) +
  labs(x="Generation", y=FITNESS_FUNC, col='') +
  scale_color_brewer(palette = 'Dark2') +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  theme(strip.text = element_text(size=12))
gg_fit <- add_window_vlines(gg_fit)
print(gg_fit)
dev.off()

# Network connections over generations
png(filename = paste(OUT_DIR, 'connections_by_gen.png', sep=''))
gg_connections_gen <- ggplot(data=evals_avg_dt, aes(generation)) +
  geom_line(aes(y=connections_mean)) +
  labs(x="Generation", y="Connections", col='') +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
gg_connections_gen <- add_window_vlines(gg_connections_gen, labels = human_numbers, scale_break_freq = 3)
print(gg_connections_gen)
dev.off()

# prediction time over gens
png(filename = paste(OUT_DIR, 'pred_time_by_gen.png', sep = ''))
evals_avg_sample <- evals_avg_dt[sample(1:nrow(evals_avg_dt), 30000)]
gg_pred_time_gen <- ggplot(data=evals_avg_sample, aes(generation)) + 
  geom_smooth(aes(y=pred_avg_time), span=0.001, fill=gsmooth_fill, method = 'loess') +
  geom_hline(yintercept = 50) +
  geom_hline(yintercept = 40) +
  geom_text(aes(3000, 40, label = 'lower limit: 40µs', vjust = -1)) +
  geom_text(aes(3000, 50, label = 'upper limit: 50µs', vjust = -1)) +
  labs(x="Generation", y="Prediction time (µs)") + 
  scale_color_brewer(palette = 'Dark2') +
  scale_x_continuous(breaks=gen_breaks, labels = human_numbers) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
gg_pred_time_gen <- add_window_vlines(gg_pred_time_gen, labels = human_numbers, scale_break_freq = 3)
print(gg_pred_time_gen)
dev.off()

# Network neurons over generations
png(filename = paste(OUT_DIR, 'neurons_by_gen.png', sep=''))
gg_neurons_gen <- ggplot(data=evals_avg_dt, aes(generation)) +
  geom_line(aes(y=neurons_mean)) +
  labs(x="Generation", y="Neurons", col='') +
  theme_minimal()
gg_neurons_gen <- add_window_vlines(gg_neurons_gen)
print(gg_neurons_gen)
dev.off()

# Network neurons and connections over generations
png(filename = paste(OUT_DIR, 'cons_neurons_by_gen.png', sep=''))
gg_cons_neurons_gen <- ggplot(data=evals_avg_dt, aes(generation)) +
  geom_line(aes(y=connections_mean, col='Connections')) +
  geom_line(aes(y=neurons_mean, col='Neurons')) +
  labs(x="Generation", y="Complexity", col='') +
  scale_color_brewer(palette = 'Dark2') +
  theme_minimal()
gg_cons_neurons_gen <- add_window_vlines(gg_cons_neurons_gen)
print(gg_cons_neurons_gen)
dev.off()

# Network connections over generations (first run)
png(filename = paste(OUT_DIR, 'connections_by_gen_single.png', sep=''))
gg_connections_gen_single <- ggplot(data=evals_single_dt, aes(generation)) +
  geom_line(aes(y=connections_mean)) +
  geom_hline(yintercept = 100) +
  geom_hline(yintercept = 150) +
  geom_text(aes(500, 100, label = 'lower limit: 100', vjust = -1)) +
  geom_text(aes(500, 150, label = 'upper limit: 150', vjust = -1)) +
  labs(x="Generation", y="Connections", col='') +
  scale_x_continuous(breaks=gen_breaks) +
  theme_minimal()
gg_connections_gen_single <- add_window_vlines(gg_connections_gen_single)
print(gg_connections_gen_single)
dev.off()

# Network neurons over generations (first run)
png(filename = paste(OUT_DIR, 'neurons_by_gen_single.png', sep=''))
gg_neurons_gen_single <- ggplot(data=evals_single_dt, aes(generation)) +
  geom_line(aes(y=neurons_mean)) +
  labs(x="Generation", y="Neurons", col='') +
  theme_minimal()
gg_neurons_gen_single <- add_window_vlines(gg_neurons_gen_single)
print(gg_neurons_gen_single)
dev.off()

# Mutation probs over generations (first)
if(exists("mutations")){
  png(filename = paste(OUT_DIR, 'mutation_probs_over_gens.png', sep=''))
  gg_muts <- ggplot(data=mutations, aes(x=generation, y=prob*100, col=mutation)) +
    geom_line(size=.8) +
    scale_color_brewer(palette = 'Paired', direction = -1) +
    labs(x='Generation', y='Probability (%)', col='Mutation') +
    scale_x_continuous(breaks=gen_breaks) +
    theme_minimal()
  gg_muts <- add_window_vlines(gg_muts)
  print(gg_muts)
  dev.off()
}

# Eval times over generations
png(filename = paste(OUT_DIR, 'times_by_gen.png', sep=''))
gg_eval_times <- ggplot(eval_times, aes(x=generation, y=time, color=state)) +
  geom_smooth(fill=gsmooth_fill) +
  labs(x='Generation', y='Time (μs)', color='Times') +
  scale_color_brewer(palette = 'Dark2') +
  theme_minimal()
gg_eval_times <- add_window_vlines(gg_eval_times)
print(gg_eval_times)
dev.off()

# Deviation | Fitness,connections,eval_time
png(filename = paste(OUT_DIR, 'fit_con_time_deviation.png', sep=''))
gg_devs <- ggplot(evals_dev, aes(x=generation)) +
  geom_smooth(aes(y=value, col=variable), fill=gsmooth_fill) +
  scale_color_brewer(palette = "Dark2", direction=-1) +
  labs(x='Generation', y='Deviation from mean', col='') +
  theme_minimal()
gg_devs <- add_window_vlines(gg_devs)
print(gg_devs)
dev.off()

# Pairs | Fitness,connections,eval_time
png(filename = paste(OUT_DIR, 'fit_con_time_pairs.png', sep=''))
ggpairs(evals_pairs, lower = list(continuous = wrap("points", alpha = 0.2, size=0.2)))
dev.off()
