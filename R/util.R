library(data.table)

setup <- function(multi_types=FALSE){
  fit_label <<- paste("Fitness ", "(", FITNESS_FUNC, ")", sep='')
  has_windows <<- exists("WINDOWS") && WINDOWS > 1
  options(digits=DIGITS)
  
  if(multi_types){
    labels_ord <<- sapply(RUN_TYPES, function(x){RUN_TYPE_LABEL[[x]]})
    evals_file_names <<- hash()
    summs_file_names <<- hash()
    windows_file_names <<- hash()
    for(type in RUN_TYPES){
      prefix = paste(RESULTS_DIR, type, sep='')
      if(RUNS > 1){
        prefix = paste(prefix, '(', 1:RUNS, ')', sep='')
      }
      
      evals_file_names[type] <- paste(prefix, '_evaluations.csv', sep='')
      if(has_windows){
        prefix_w_window = CJ(prefix, 0:(WINDOWS-1), sorted = FALSE)[, paste(V1, '(', V2, ')', sep ="")]
        summs_file_names[type] <- paste(prefix_w_window, '_summary.json', sep='')
        windows_file_names[type] <- paste(prefix, '_windows.csv', sep='') 
      }else{
        summs_file_names[type] <- paste(prefix, '_summary.json', sep='')
      }
    }
  }
  else{
    prefix = paste(RESULTS_DIR, RUN_PREFIX, sep='')
    if(RUNS > 1){prefix = paste(prefix, '(', 1:RUNS, ')', sep='')}
    evals_file_names <<- paste(prefix, '_evaluations.csv', sep='')
    windows_file_names <<- paste(prefix, '_windows.csv', sep='')
    if(has_windows){
      prefix_w_window = CJ(prefix, 0:(WINDOWS-1), sorted = FALSE)[, paste(V1, '(', V2, ')', sep ="")]
      summs_file_names <<- paste(prefix_w_window, '_summary.json', sep='')
      windows_file_names <<- paste(prefix, '_windows.csv', sep='') 
    }else{
      summs_file_names <<- paste(prefix, '_summary.json', sep='')
    }
  }
}

read_evaluations <- function(evals_file_names){
  evals_dt = rbindlist(lapply(1:length(evals_file_names), function(i){
    run_dt = data.table(read.csv(file=evals_file_names[i], header=TRUE, sep=','))
    run_dt$eval_time = run_dt$build_time + run_dt$pred_time + run_dt$fit_time
    run_dt$run = rep(i, nrow(run_dt))
    run_dt
  }))
}

group_evals <- function(evals_dt){
  # First average each run
  evals_run_avg = evals_dt[ , .(window=round(median(window)), fitness_mean = mean(fitness), fitness_best = max(fitness),
                                fitness_test_mean = mean(fitness_test), fitness_test_best = fitness_test[which.max(fitness)],
                                neurons_mean = mean(neurons), neurons_max = max(neurons), neurons_best = neurons[which.max(fitness)],
                                connections_mean = mean(connections), connections_max = max(connections), connections_best = connections[which.max(fitness)],
                                build_time=mean(build_time), pred_time=mean(pred_time), pred_avg_time=mean(pred_avg_time), 
                                fit_time=mean(fit_time), eval_time=mean(build_time+pred_time+fit_time),
                                run_time = mean(run_time)), by = list(run, generation)]
  
  # Count how many times each generation occurs
  generation_count = table(evals_run_avg$generation)
  # Crop outlier generations that appear in less than 80% of runs
  evals_run_avg = evals_run_avg[evals_run_avg$generation %in% names(generation_count)[generation_count>=0.8*RUNS],]
  
  # Then return the average of every run's average
  evals_avg = evals_run_avg[ , .(window=round(median(window)), fitness_mean = mean(fitness_mean), fitness_best = mean(fitness_best),
                                 fitness_test_mean = mean(fitness_test_mean), fitness_test_best = mean(fitness_test_best),
                                 neurons_mean = mean(neurons_mean), neurons_max = mean(neurons_max), neurons_best = mean(neurons_best),
                                 connections_mean = mean(connections_mean), connections_max = mean(connections_max), connections_best = mean(connections_best),
                                 build_time=mean(build_time), pred_time=mean(pred_time), pred_avg_time=mean(pred_avg_time), 
                                 fit_time=mean(fit_time), eval_time=mean(build_time+pred_time+fit_time),
                                 run_time = mean(run_time)), by = generation]
}

get_evals_sample <- function(evals_dt){
  pop_size = nrow(evals_dt[generation==0]) / RUNS
  sample_size = nrow(evals_dt)%/%pop_size
  evals_dt[sample(.N, sample_size)]
}

read_summaries <- function(summs_file_names){
  if(length(summs_file_names) == 0)
    return(data.table())
  
  run_summaries = lapply(1:length(summs_file_names), function(i){
    summary_json = fromJSON(file=summs_file_names[i])
    summary_dt = data.table(run=i, time_ea=chron(time=summary_json$ea_time), time_eval=chron(time=summary_json$eval_time),
                            time_total=chron(time=summary_json$run_time), generations=summary_json$generations, train_fit=summary_json$best$fitness,
                            test_fit=summary_json$best$fitness_test, neurons=summary_json$best$neurons_qty, connections=summary_json$best$connections_qty)
  })
  run_summaries = rbindlist(run_summaries)
}

group_summaries <- function(summaries_dt, group_by=''){
  summaries_dt[, .(time_ea=mean(time_ea), time_eval=mean(time_eval), time_total=mean(time_total), generations=round(mean(generations)),
                   train_fit=mean(train_fit), test_fit=mean(test_fit), neurons=round(mean(neurons)), connections=round(mean(connections))),
               by = group_by]
}

read_windows <- function(windows_file_names){
  windows_dt = rbindlist(lapply(windows_file_names, function(file_name){
    summary_dt = data.table(read.csv(file=file_name, header=TRUE, sep=','))
  }))
  windows_dt$window_factor = factor(windows_dt$window, ordered = TRUE)
  windows_dt
}

group_windows <- function(windows_dt, group_by="window"){
  windows_avg_dt <- windows_dt[, .(window_factor=first(window_factor), begin_date=first(begin_date), end_date=first(end_date), generations=mean(generations), 
                                   run_time=mean(run_time), eval_time=mean(eval_time), ea_time=mean(ea_time),
                                   train_size=mean(train_size), train_positives=mean(train_positives), train_negatives=mean(train_negatives),
                                   test_size=mean(test_size), test_positives=mean(test_positives), test_negatives=mean(test_negatives),
                                   train_fitness=mean(train_fitness), test_fitness=mean(test_fitness), 
                                   best_neurons=round(mean(best_neurons)), best_connections=round(mean(best_connections))), by = group_by]
}

read_windows_or_summaries <- function(multi_types=FALSE){
  if(multi_types){
    if(!has_windows){
      summaries_avg_dt <<- rbindlist(lapply(RUN_TYPES, function(type){
        run_type_summaries = group_summaries(read_summaries(summs_file_names[[type]]))
        run_type_summaries$run_type = list(RUN_TYPE_LABEL[[type]])
        run_type_summaries
      }))
      summaries_avg_dt$run_type <- factor(summaries_dt$run_type, levels=rev(labels_ord), ordered = TRUE)
    } else {
      windows_dt <<- rbindlist(lapply(RUN_TYPES, function(type){
        run_type_windows = read_windows(windows_file_names[[type]])
        run_type_windows$run_type = rep(RUN_TYPE_LABEL[[type]], nrow(run_type_windows))
        run_type_windows
      }))
      windows_dt$run_type <- factor(windows_dt$run_type, levels=rev(labels_ord), ordered = TRUE)
      windows_avg_dt <<- group_windows(windows_avg_dt, group_by = c("run_type", "window"))
      windows_gen_splits <<- windows_avg_dt$generations[1:(length(windows_avg_dt$generations)-1)]
    }
  } else{
    if(!has_windows){
      summaries_dt <<- read_summaries(summs_file_names)
      summaries_avg_dt <<- group_summaries(summaries_dt)
    } else{
      windows_dt <<- read_windows(windows_file_names)
      windows_avg_dt <<- group_windows(windows_dt)
      windows_gen_splits <<- windows_avg_dt$generations[1:(length(windows_avg_dt$generations)-1)]
    }
  }
}

move_to_first <- function(data, move) {
  data[c(move, setdiff(names(data), move))]
}








