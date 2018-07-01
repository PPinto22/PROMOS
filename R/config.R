# Common configs
RUNS <<- 4
WINDOWS <<- 10
RESULTS_DIR <<- '../results/example/'
OUT_DIR <<- 'out/example/'
FITNESS_FUNC <<- 'AUC'
DIGITS <<- 5

# For evolution.R
RUN_PREFIX <<- 'neat_example'

# For evolution_comparison.R 
library(hash)
RUN_TYPES <<- c('neat_example1', 'neat_example2')
RUN_TYPE_LABEL <<- hash(keys=RUN_TYPES, values=c('Example 1', 'Example2'))
SERIES_LABEL <<- 'Run'