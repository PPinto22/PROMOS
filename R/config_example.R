# For evolution.R
RUNS <<- 4
WINDOWS <<- 10
RESULTS_DIR <<- '../results/example'
RUN_PREFIX <<- 'neat_example'

# For evolution_comparison.R
RUN_TYPES <<- c('neat_example1', 'neat_example2')
RUNS <<- c(4, 4)
WINDOWS <<- c(10, 10)
RESULTS_DIR <<- c('../results/example1', '../results/example2')
RUN_TYPE_LABELS <<- c('Example 1', 'Example 2')
SERIES_LABEL <<- 'Run'

# Common configs
OUT_DIR <<- 'out/example'
FITNESS_FUNC <<- 'AUC'
DIGITS <<- 5
