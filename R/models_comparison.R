# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
source('util.R')
source('config.R')

MODELS_FILE <- 'out/mlp_online.csv'

models_dt <- data.table(read.csv(file=MODELS_FILE, header=TRUE, sep=','))

setup(multi_types=T)
read_windows_or_summaries()

models_crop_dt <- models_dt[, c("window", "auc", "time")]
models_crop_dt$algorithm <- toupper(models_dt$algorithm)
models_crop_dt$encoding <- toupper(models_dt$encoding)
models_crop_dt <- models_crop_dt[ , .(auc=mean(auc), time=mean(time)), by=list(algorithm, encoding, window)]
models_avg_dt <- models_crop_dt[ , .(auc=mean(auc), time=sum(time)), by=list(algorithm, encoding)]

neuro_crop_dt <- windows_avg_dt[, c("run_type", "window", "test_fitness", "eval_time", "ea_time")]
neuro_crop_dt$window <- neuro_crop_dt$window+1
neuro_crop_dt$algorithm <- sapply(neuro_crop_dt$run_type, function(x){ strsplit(as.character(x), " ")[[1]][1] })
neuro_crop_dt$encoding <- sapply(neuro_crop_dt$run_type, function(x){ strsplit(as.character(x), " ")[[1]][2] })
neuro_crop_dt$run_type <- NULL
neuro_crop_dt$auc <- neuro_crop_dt$test_fitness
neuro_crop_dt$test_fitness <- NULL
neuro_crop_dt$time <- neuro_crop_dt$ea_time + neuro_crop_dt$eval_time
neuro_crop_dt$ea_time <- NULL
neuro_crop_dt$eval_time <- NULL
neuro_avg_dt <- neuro_crop_dt[ , .(auc=mean(auc), time=sum(time)), by = list(algorithm, encoding)]

comp_windows_dt <- rbind(models_crop_dt, neuro_crop_dt)
# comp_windows_dt$encoding <- factor(comp_windows_dt$encoding, levels = c('RAW', 'IDF', 'CP'))
# comp_windows_dt$algorithm <- factor(comp_windows_dt$algorithm, levels = c('NEAT', 'HyperNEAT', 'LR'))
# comp_windows_dt <- comp_windows_dt[type %in% c("LR PCP", "NEAT PCP")]

comp_avg_dt <- rbind(models_avg_dt, neuro_avg_dt)
# write.table(comp_avg_dt, file='out/models/models_comp_window_avg.csv', sep=',', row.names = FALSE)


ggplot(comp_windows_dt, aes(x=window, col=algorithm)) +
  facet_wrap(~encoding) +
  geom_line(aes(y = auc)) +
  geom_point(aes(y = auc)) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks=seq(1,10), minor_breaks = NULL) +
  scale_linetype_manual(NULL, values = 1:length(comp_windows_dt$algorithm)) +
  scale_shape_manual(NULL, values = 1:length(comp_windows_dt$algorithm)) +
  labs(x="Window", y="AUC", col=NULL) +
  theme_minimal()

ggplot(comp_windows_dt, aes(x=window, col=algorithm, linetype=algorithm, shape=algorithm)) +
  facet_wrap(~encoding) +
  geom_line(aes(y = time)) +
  geom_point(aes(y = time)) +
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks=seq(1,10), minor_breaks = NULL) +
  scale_linetype_manual(NULL, values = 1:length(comp_windows_dt$algorithm)) +
  scale_shape_manual(NULL, values = 1:length(comp_windows_dt$algorithm)) +
  labs(x="Window", y="Time (min)", col=NULL) +
  theme_minimal()

# Median -> average
neuro_median <- windows_dt[, c("run_type", "run", "window", "test_fitness", "eval_time", "ea_time", "best_neurons", "best_connections")]
neuro_median$window <- neuro_median$window+1
neuro_median$mode <- sapply(neuro_median$run_type, function(x){ strsplit(as.character(x), " ")[[1]][1] })
neuro_median$algorithm <- sapply(neuro_median$run_type, function(x){ strsplit(as.character(x), " ")[[1]][2] })
neuro_median$encoding <- sapply(neuro_median$run_type, function(x){ strsplit(as.character(x), " ")[[1]][3] })
neuro_median$run_type <- NULL
neuro_median$auc <- neuro_median$test_fitness
neuro_median$test_fitness <- NULL
neuro_median$time <- neuro_median$ea_time + neuro_median$eval_time
neuro_median$ea_time <- NULL
neuro_median$eval_time <- NULL
neuro_median <- neuro_median[, .(auc=median(auc), time=median(time),
                                 best_neurons=median(best_neurons), best_connections=median(best_connections)),
                             by = list(run, mode, algorithm, encoding)]
neuro_median_avg <- neuro_median[, .(auc=mean(auc), time=mean(time),
                                     best_neurons=mean(best_neurons), best_connections=mean(best_connections)),
                                 by = list(mode, algorithm, encoding)]
