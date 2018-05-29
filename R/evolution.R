# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# This script reads result files from multiple executions named as, e.g., 'neat_1K(N)_evaluations.csv', where 
# neat_1K is a prefix that identifies the run type, and 'N' is an integer ranging from 1 to the number of runs

# install.packages("hash")
# install.packages("ggpubr")
# install.packages("ggplot2")
# install.packages("data.table")
# install.packages("rjson")
# install.packages("chron")
library(hash)
library(ggplot2)
library(ggpubr)
library(data.table)
library(rjson)
library(chron)

# ---- CONFIGURATION ----
RUNS <- 30
RESULTS_DIR <- '../results/NEAT/samples_30runs/'
RUN_TYPES <- c('neat_ALL', 'neat_10K', 'neat_1K', 'neat_100') # These are the prefixes of the result files
RUN_TYPE_LABEL <- hash(keys=RUN_TYPES, values=c('All (165K)', '10 000', '1 000', '100'))
SERIES_LABEL <- 'Sample size'
FITNESS_FUNC <- 'AUC'
OUT_DIR <- 'out/samples_30/'
DIGITS <- 5

# ---- SETUP ----
options(digits=DIGITS)
labels_ord = sapply(RUN_TYPES, function(x){RUN_TYPE_LABEL[[x]]})
# Map, for each run type (i.e., each prefix in PREFIX), a list of all respective data files
evals_file_names <- hash()
summs_file_names <- hash()
for(type in RUN_TYPES){
  prefix = paste(RESULTS_DIR, type, sep='')
  if(RUNS > 1)
    prefix = paste(prefix, '(', 1:RUNS, ')', sep='')
  
  evals_file_names[type] <- paste(prefix, '_evaluations.csv', sep='')
  summs_file_names[type] <- paste(prefix, '_summary.json', sep='')
}

# Read all evaluations.csv files and store the averages of all runs
evals_dt <- rbindlist(lapply(RUN_TYPES, function(type){
  # Read all runs of the current type
  run_type_dts = lapply(evals_file_names[[type]], function(file_name){
    run_dt = data.table(read.csv(file=file_name, header=TRUE, sep=','))
    # Group by generation
    run_dt = run_dt[ , .(fitness.mean = mean(fitness), fitness.max = max(fitness),
                         neurons.mean = mean(neurons), neurons.max = max(neurons), neurons.best = neurons[which.max(fitness)],
                         connections.mean = mean(connections), connections.max = max(connections), connections.best = connections[which.max(fitness)],
                         time = mean(run_minutes)), by = generation]
  })
  # Join all runs into a single data.table
  run_type_dts = rbindlist(run_type_dts)

  # Count how many times each generation occurs
  generation_count = table(run_type_dts$generation)
  # Crop outlier generations that appear in less than 80% of runs
  run_type_dts = run_type_dts[run_type_dts$generation %in% names(generation_count)[generation_count>=0.8*RUNS],]

  # Get the average of run_type_dts
  type_avg_dt = run_type_dts[, .(fitness.mean = mean(fitness.mean), fitness.max = mean(fitness.max),
                                 neurons.mean = mean(neurons.mean), neurons.max = mean(neurons.max), neurons.best = mean(neurons.best),
                                 connections.mean = mean(connections.mean), connections.max = mean(connections.max), connections.best = mean(connections.best),
                                 time = mean(time)), by = generation]
  # Add "run_type" column
  run_type = rep(RUN_TYPE_LABEL[[type]], nrow(type_avg_dt))
  type_avg_dt = cbind(type_avg_dt, run_type)
  return(type_avg_dt)
}))
evals_dt$run_type <- factor(evals_dt$run_type, levels=rev(labels_ord), ordered = TRUE)

# Read summaries
summaries_dt <- rbindlist(lapply(RUN_TYPES, function(type){
  run_type_summaries = lapply(summs_file_names[[type]], function(file_name){
    summary_json = fromJSON(file=file_name)
    summary_dt = data.table(run_type=RUN_TYPE_LABEL[[type]], time_ea=chron(time=summary_json$ea_time), time_eval=chron(time=summary_json$eval_time),
                             time_total=chron(time=summary_json$run_time), generations=summary_json$generations, train_fit=summary_json$best$fitness,
                             test_fit=summary_json$best$fitness_test, neurons=summary_json$best$neurons_qty, connections=summary_json$best$connections_qty)
  })
  run_type_summaries = rbindlist(run_type_summaries)
}))
summaries_dt$run_type <- factor(summaries_dt$run_type, levels=rev(labels_ord), ordered = TRUE)

summaries_avg_dt <- summaries_dt[, .(time_ea=mean(time_ea), time_eval=mean(time_eval), time_total=mean(time_total), generations=round(mean(generations)),
                                     train_fit=mean(train_fit), train_fit_ci=list(t.test(train_fit)$conf.int),
                                     test_fit=mean(test_fit), test_fit_ci=list(t.test(test_fit)$conf.int),
                                     neurons=round(mean(neurons)), neurons_ci=list(t.test(neurons)$conf.int),
                                     connections=round(mean(connections)), connections_ci=list(t.test(connections)$conf.int)),
                                     by = run_type]

# ---- OUTPUTS ----
# Create OUT_DIR
dir.create(file.path(OUT_DIR), recursive=TRUE, showWarnings=FALSE)

# -- Summary table --
summaries_avg_str_dt <- summaries_avg_dt[, !c('train_fit', 'test_fit', 'neurons', 'connections')]
ci2str <- function(ci){paste('[', format(ci[1], digits=DIGITS), ', ', format(ci[2], digits=DIGITS), ']', sep='')}
original_colnames <- copy(colnames(summaries_avg_str_dt))
ci_col_names <- c('train_fit_ci', 'test_fit_ci', 'neurons_ci', 'connections_ci')
summaries_avg_str_dt[, (paste(ci_col_names,'TEMP')):=lapply(.SD, function(x) sapply(x, ci2str)), .SDcols=ci_col_names][,(ci_col_names):=NULL]
setnames(summaries_avg_str_dt, paste(ci_col_names,'TEMP'), ci_col_names)
setcolorder(summaries_avg_str_dt, original_colnames)
write.table(summaries_avg_str_dt, file=paste(OUT_DIR, 'summary.csv', sep=''), row.names = FALSE, sep=',',
          col.names = c(SERIES_LABEL, 'Time (EA)', 'Time (Evaluation)', 'Time (Total)', 'Generations', 'Fitness (Train)', 'Fitness (Test)', 'Neurons', 'Connections'))

# -- P-values table --
test_fit_pvalues <- as.data.table(lapply(labels_ord, function(run_type1){
  sapply(labels_ord, function(run_type2){
    format(t.test(summaries_dt[run_type==run_type1, train_fit], summaries_dt[run_type==run_type2, train_fit])$p.value, digits=1)
  })
}))
test_fit_pvalues[upper.tri(test_fit_pvalues, diag=TRUE)] <- NA
colnames(test_fit_pvalues) <- labels_ord
rownames(test_fit_pvalues) <- labels_ord
write.table(test_fit_pvalues, file=paste(OUT_DIR, 'pvalues_fitness.csv', sep=''), sep=',', row.names=TRUE, col.names=NA, na = '-')

# -- Graphs --
# Boxplot best test fitness by sample size
png(filename = paste(OUT_DIR, 'fitness_best_bp.png', sep=''))
gg_best_testfit <- ggplot(data=summaries_dt, aes(x=run_type, y=test_fit)) +
  geom_boxplot() +
  labs(x=SERIES_LABEL, y=paste("Fitness ", "(", FITNESS_FUNC, ")", sep='')) +
  theme_minimal()
gg_best_testfit
dev.off()

# Boxplot #connections of the best individual by sample size
png(filename = paste(OUT_DIR, 'connections_best_bp.png', sep=''))
gg_best_connections <- ggplot(data=summaries_dt, aes(x=run_type, y=connections)) +
  geom_boxplot() +
  labs(x=SERIES_LABEL, y=paste("Connections", sep='')) +
  theme_minimal()
gg_best_connections
dev.off()

# Plot mean fitness over time
png(filename = paste(OUT_DIR, 'fitness_mean.png', sep=''))
gg_meanfit <- ggplot(data=evals_dt, aes(time)) + 
  geom_smooth(aes(y=fitness.mean, col=run_type), method='loess') +
  labs(x="Run time (min)", y=paste("Fitness ", "(", FITNESS_FUNC, ")", sep=''), col=SERIES_LABEL) +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) +
  theme_minimal()
gg_meanfit
dev.off()

# Plot max fitness over time
png(filename = paste(OUT_DIR, 'fitness_max.png', sep=''))
gg_maxfit <- ggplot(data=evals_dt, aes(time)) + 
  geom_smooth(aes(y=fitness.max, col=run_type), method='loess') +
  labs(x="Run time (min)", y=paste("Fitness ", "(", FITNESS_FUNC, ")", sep=''), col="Sample size") +
  scale_y_continuous(limits=c(0.49, 1.0), breaks=seq(0.5,1,0.05)) + 
  theme_minimal()
gg_maxfit
dev.off()

# Plot generations over time
png(filename = paste(OUT_DIR, 'generations.png', sep=''))
gg_generations <- ggplot(data=evals_dt, aes(time)) + 
  geom_smooth(aes(y=generation, col=run_type), method='loess') +
  labs(x="Run time (min)", y="Generations", col=SERIES_LABEL) + 
  theme_minimal()
gg_generations
dev.off()

# Plot mean complexity (connections) over time
png(filename = paste(OUT_DIR, 'connections_mean.png', sep=''))
gg_connections <- ggplot(data=evals_dt, aes(time)) + 
  geom_line(aes(y=connections.mean, col=run_type)) +
  labs(x="Run time (min)", y="Connections", col=SERIES_LABEL) + 
  theme_minimal()
gg_connections
dev.off()

# Plot complexity (connections) of the best individual over time
png(filename = paste(OUT_DIR, 'connections_best.png', sep=''))
gg_connections <- ggplot(data=evals_dt, aes(time)) + 
  geom_line(aes(y=connections.best, col=run_type)) +
  labs(x="Run time (min)", y="Connections", col=SERIES_LABEL) + 
  theme_minimal()
gg_connections
dev.off()
