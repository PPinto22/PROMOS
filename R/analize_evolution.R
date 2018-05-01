data <- read.csv(file='../results/neat_2018-05-01_18:57:40_evaluations.csv', header=TRUE, sep=',')

gens_max <- aggregate(. ~ generation, data = data, max)

plot(gens_max$generation, gens_max$fitness, xlab = "Generations", ylab = "Fitness (AUC)")
plot(gens_max$generation, gens_max$neurons)
plot(gens_max$generation, gens_max$connections)
