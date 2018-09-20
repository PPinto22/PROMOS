# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(data.table)
source('util.R')
source('config.R')

MODELS_FILE <- 'out/models/xgb_windows.csv'

models_dt <- data.table(read.csv(file=MODELS_FILE, header=TRUE, sep=','))

setup(multi_types=T)
read_windows_or_summaries()

models_crop_dt <- models_dt[, c("window", "auc", "time")]
models_crop_dt$time <- models_crop_dt$time/60
models_crop_dt$type <- paste(models_dt$algorithm, models_dt$encoding)
models_crop_dt$type <- toupper(models_crop_dt$type)
models_avg_dt <- models_crop_dt[ , .(auc=mean(auc), time=sum(time)), by=type]

neuro_crop_dt <- windows_avg_dt[, c("run_type", "window", "test_fitness", "eval_time", "ea_time")]
neuro_crop_dt$window <- neuro_crop_dt$window+1
neuro_crop_dt$type <- neuro_crop_dt$run_type 
neuro_crop_dt$run_type <- NULL
neuro_crop_dt$auc <- neuro_crop_dt$test_fitness
neuro_crop_dt$test_fitness <- NULL
neuro_crop_dt$time <- neuro_crop_dt$ea_time + neuro_crop_dt$eval_time
neuro_crop_dt$ea_time <- NULL
neuro_crop_dt$eval_time <- NULL
neuro_avg_dt <- neuro_crop_dt[ , .(auc=mean(auc), time=sum(time)), by=type]

comp_windows_dt <- rbind(models_crop_dt, neuro_crop_dt)
# comp_windows_dt <- comp_windows_dt[type %in% c("LR PCP", "NEAT PCP")]

comp_avg_dt <- rbind(models_avg_dt, neuro_avg_dt)
# write.table(comp_avg_dt, file='out/models/models_comp_window_avg.csv', sep=',', row.names = FALSE)

ggplot(comp_windows_dt, aes(x=window, col=type)) +
  geom_line(aes(y = auc)) +
  scale_color_brewer(palette = "Set2") +
  scale_x_continuous(breaks=seq(1,10), minor_breaks = NULL) +
  labs(x="Window", y="AUC", col=NULL) +
  theme_minimal()

ggplot(comp_windows_dt, aes(x=window, col=type)) +
  geom_line(aes(y = time)) +
  scale_color_brewer(palette = "Set2") +
  scale_x_continuous(breaks=seq(1,10), minor_breaks = NULL) +
  labs(x="Window", y="Time (min)", col=NULL) +
  theme_minimal()

